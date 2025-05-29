import numpy as np
import pandas as pd

# Libraries for advanced data structures
from collections import defaultdict, Counter

# Fast Fourier Transform from SciPy
from scipy.fft import fft

# Progress bar for loops
from tqdm import tqdm

# Joblib for saving models or large data files efficiently
import joblib

# Libraries for Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib import cm
import seaborn as sns
import plotly.express as px

# Scikit-learn Model Selection
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

# LightGBM libraries for building models
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor

# Scikit-learn Metrics for Model Evaluation
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Others
from math import sqrt
import re
import gc # Garbage collector

# For geolocalisation and distances
import json
from geopy.distance import geodesic
import geopandas as gpd
from haversine import haversine_vector, Unit

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# ==================================================================================================
# Directories, links and file paths

# Directory containing files
filedir = "../Files/"

# --------------------------------------------------------------------------------------------------
# OUTPUTS FROM NOTEBOOKS

# Directory where notebook outputs will be saved
output_dirpath = filedir + "Outputs/"

# Filepath for prepared outages data (output of notebook "1a. Create CSV")
outages_data_filepath = output_dirpath + "df.csv"

# Path to csv-dataframe containing storms data (output of notebook "1b. STORMS")
# Source: provided by the competition host on ThinkOnWard.
storms_data_filepath = output_dirpath + "df_storms_by_fips.csv"

# Path to csv-dataframe containing daily weather information (output of notebook "1c. Prepare daily weather (external dataset)")
# Source: https://www.ncei.noaa.gov/data/global-summary-of-the-day/
daily_weather_filepath = output_dirpath + "daily_weather_info.csv"

# Path to the dictionnary containing the closest station to each fips (output of notebook "1c. Prepare daily weather (external dataset)")
filepath_dict_closest_station = output_dirpath + "dict_closest_station_to_fips.joblib"

# Path to the dataset we will train the model on (output of notebook "1c. Prepare daily weather (external dataset)")
filepath_to_train_dataset = output_dirpath + "df_hourly_outages.csv"

# Path to the dataset we will infer the model on (output of notebook "2. Prepare Dataset")
filepath_to_inference_dataset = output_dirpath + "df_hourly_outages.csv"

# Path to trained model (output of notebook "3. Training")
filepath_trained_model = output_dirpath + "lgbm_model.joblib"

# --------------------------------------------------------------------------------------------------
# EXTERNAL DATA

# Shapefile for geopandas to plot USA map
# Source: https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_20m.zip
gpd_url = filedir + "External_Data/Demographic/cb_2021_us_county_20m"

# Path to csv-dataframe containing latitude and longitudes of all counties
# Source: https://gist.github.com/russellsamora/12be4f9f574e92413ea3f92ce1bc58e6
path_to_latitude_longitude_file = filedir + "External_Data/Demographic/us_county_latlng.csv"

# Path to txt file containing population data
# Source: https://seer.cancer.gov/popdata/download.html
population_filepath = filedir + "External_Data/Demographic/us.1969_2023.20ages.adjusted.txt"

# Directory to daily weather information (temperature & wind speed)
# Source: https://www.ncei.noaa.gov/data/global-summary-of-the-day/
dirpath_external_weather_data = filedir + "External_Data/Daily_Weather/global-summary-of-the-day/"

# -----------------------------------------------------------------------------------------------------
# Custom color map for better vizualisation (blue to red)
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#e5eef8', '#A7C7E7', '#E85E4C'])
custom_cmap_reverted = LinearSegmentedColormap.from_list('custom_cmap', ['#E85E4C', '#A7C7E7', '#e5eef8'])

# -----------------------------------------------------------------------------------------------------
# State abbreviations
state_abbreviations = {
                        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
                        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
                        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN',
                        'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
                        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
                        'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
                        'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
                        'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Puerto Rico': 'PR', 'Rhode Island': 'RI',
                        'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
                        'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI',
                        'Wyoming': 'WY', 'United States Virgin Islands': 'VI',
                     }

# Reverse state abbreviations
reverse_state_abbreviations = {v:k for k, v in state_abbreviations.items()}

# Dict to get state from the first 2 caracters of a 5-caracter fips-code
state_by_fips = {'00': 'XX',
                 '01': 'AL',
                 '02': 'AK',
                 '04': 'AZ',
                 '05': 'AR',
                 '06': 'CA',
                 '08': 'CO',
                 '09': 'CT',
                 '10': 'DE',
                 '11': 'DC',
                 '12': 'FL',
                 '13': 'GA',
                 '15': 'HI',
                 '16': 'ID',
                 '17': 'IL',
                 '18': 'IN',
                 '19': 'IA',
                 '20': 'KS',
                 '21': 'KY',
                 '22': 'LA',
                 '23': 'ME',
                 '24': 'MD',
                 '25': 'MA',
                 '26': 'MI',
                 '27': 'MN',
                 '28': 'MS',
                 '29': 'MO',
                 '30': 'MT',
                 '31': 'NE',
                 '32': 'NV',
                 '33': 'NH',
                 '34': 'NJ',
                 '35': 'NM',
                 '36': 'NY',
                 '37': 'NC',
                 '38': 'ND',
                 '39': 'OH',
                 '40': 'OK',
                 '41': 'OR',
                 '42': 'PA',
                 '44': 'RI',
                 '45': 'SC',
                 '46': 'SD',
                 '47': 'TN',
                 '48': 'TX',
                 '49': 'UT',
                 '50': 'VT',
                 '51': 'VA',
                 '53': 'WA',
                 '54': 'WV',
                 '55': 'WI',
                 '56': 'WY'}

# -----------------------------------------------------------------------------------------------------

import os, psutil  
def memory_usage():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'Memory (GB): ' + str(np.round(memory_use, 2))

# -----------------------------------------------------------------------------------------------------

def remove_non_alphabetic_char(input_string):
    return re.sub(r'[^a-zA-Z]', '', input_string)

# -----------------------------------------------------------------------------------------------------

def find_color_neighbours(k_) :
    """
    Returns a color code corresponding to a category keyword found in the input string.

    Parameters: k_ (str): A string containing a category identifier, such as ranking ('1st', '2nd', etc.) or status keywords ('by_state'...).
    Returns: str: A hexadecimal or named color string representing the associated category.
    """
    # Lower
    k = k_.lower()
    if '1st' in k : return '#722118'
    elif '2nd' in k : return '#c23728'
    elif '3rd' in k : return '#F99949'
    elif '4th' in k : return '#E99B8B'
    elif 'state' in k : return 'gray'
    elif 'all_neighbours' in k : return '#c54f4d'
    else : return 'yellow'

# -----------------------------------------------------------------------------------------------------

def reduce_memory_usage(df, verbose=True):
    """
    Downcasts numerical columns in a DataFrame to more memory-efficient types without loss of information.
    
    Parameters: df (pd.DataFrame): Input pandas DataFrame to optimize. verbose (bool, optional): If True, prints memory usage before and after optimization. Default is True.
    Returns: pd.DataFrame: Optimized DataFrame with reduced memory usage.
    """
    
    start_mem = round(df.memory_usage().sum() / 1024**2, 1)
    if verbose: print('BEFORE: Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype.name
        if ((col_type != 'datetime64[ns]') & (col_type != 'category') & (col_type != 'object')):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    pass
    mem_usg = round(df.memory_usage().sum() / 1024**2 , 1)
    if verbose: print("AFTER: Memory usage became: ",mem_usg," MB")
    
    return df

# -----------------------------------------------------------------------------------------------------

def extract_date_features(df, time_col, drop=False) :
    """
    Extracts common date-related features from a datetime column and appends them to the DataFrame.

    Parameters: - df (pd.DataFrame): Input DataFrame containing the datetime column.
                - time_col (str): Name of the column containing datetime values.
                - drop (bool, optional, default = False): If True, drops the original datetime column after feature extraction.
    
    Returns: pd.DataFrame: DataFrame enriched with new date-related features.
    """
    # Force datetime type
    df[time_col] = pd.to_datetime(df[time_col])

    # Date features (+Optimize dtypes to save memory)
    df["date"]        = df[time_col].astype(str).apply(lambda x : x[:10])
    #df["year"]        = df[time_col].dt.year.astype('int16')
    df["month"]       = df[time_col].dt.month
    #df["day"]         = df[time_col].dt.day
    df["week"]        = df[time_col].dt.isocalendar().week
    df["dayofweek"]   = df[time_col].dt.dayofweek.astype('int8')
    df["hour"]        = df[time_col].dt.hour.astype('int8')
    #df["minute"]      = df[time_col].dt.minute.astype('int8')
    #df["period"]      = df[time_col].astype(str).apply(lambda x : x[5:10])
    #df["hour_minute"] = df[time_col].astype(str).apply(lambda x : x[11:16])

    # Drop time_col
    if drop:
        df = df.drop(columns = [time_col])
        
    # Return
    return df

# -----------------------------------------------------------------------------------------------------

def aggregate_df_hourly(df):
    """
    Aggregates outages to an hourly level by taking the maximum number of outages per hour for each location (fips) and date.
    
    Parameters: 
        - df (pd.DataFrame): Input DataFrame containing at least the columns 'fips', 'county', 'state', 'date', 'hour', and 'outages'.
    
    Returns: 
        pd.DataFrame: Hourly-aggregated DataFrame sorted by 'fips', 'date', and 'hour'.
    """
    
    # Groupby columns : [x for x in df.columns if x not in ['run_start_time', 'minute', 'outages']]
    cols = ['fips', 'county', 'state', 'date', 'hour']

    # Aggregate the 'customers_out' (or 'outages') to the max by hour
    df = df.groupby(cols)['outages'].max().reset_index()

    # Sort df
    df = df.sort_values(by = ['fips', 'date', 'hour']).reset_index(drop=True)
    
    # Return
    return df

# -----------------------------------------------------------------------------------------------------
    
def ensure_df_has_one_row_per_hour(df) :
    """
    Ensures the DataFrame has one row per hour for each 'fips' code between the minimum and maximum dates in the data.
    
    Parameters: 
        - df (pd.DataFrame): Input DataFrame containing at least the columns 'fips', 'date', 'hour', and 'outages'.
    
    Returns: 
        pd.DataFrame: A time-complete DataFrame with one row per hour per 'fips', filling missing combinations with NaNs.
    """

     # Date min & max
    date_min, date_max = df['date'].min(), df['date'].max()
    print(f"Date min : {str(date_min)}.")
    print(f"Date max : {str(date_max)}.")
    
    # Create a df with all dates, every hour
    df_all_dates = pd.DataFrame({'date_timestamp': pd.date_range(start=date_min, end=date_max, freq='H')})

    # Extract date (string, day-level) and hour
    df_all_dates = extract_date_features(df_all_dates, time_col='date_timestamp', drop=True)

    # Cross join to have one row per hour for each fips
    df_all_dates = pd.merge(df[["fips"]].drop_duplicates(),
                            df_all_dates,
                            how='cross')

    # Finally, we join with df
    merge_keys = ["fips", "date", "hour"]
    df = pd.merge(df_all_dates,
                  df[merge_keys + ["outages"]],
                  how='left',
                  on = merge_keys,
                 )
    
    # Fill NaN valuesif possible, the value for a row =H becomes the mean of H-1 et H+1. Otherwise, we fill it with 0.
    #df = complete_missing_values(df)

    # Sort df
    df = df.sort_values(by = ['fips', 'date', 'hour']).reset_index(drop=True)

    # Return
    return df

# -----------------------------------------------------------------------------------------------------

def complete_missing_values(df) :
    """
    Fills missing 'outages' values using the average of adjacent hourly values per 'fips' when available, otherwise fills with 0.
    
    Parameters:
        - df (pd.DataFrame): Input DataFrame sorted by 'fips', 'date', and 'hour', containing a column 'outages' with possible NaNs.
    
    Returns:
        pd.DataFrame: DataFrame with missing 'outages' values filled.
    """

    # Make sure df is sorted
    df = df.sort_values(by = ['fips', 'date', 'hour']).reset_index(drop=True)
    
    # Fill the NaN with mean values of adjacents rows when possible (from Row-1 and Row+1)
    outages_mean = ((df.groupby(['fips'])['outages'].shift(1) + df.groupby(['fips'])['outages'].shift(-1)) / 2 )
    mask = (df['outages'].isna())
    df.loc[mask, 'outages'] = outages_mean.loc[mask]
    
    # Otherwise, fill with 0
    df['outages'] = df['outages'].fillna(0)

    # Return
    return df

# -----------------------------------------------------------------------------------------------------
   
def plot_state_on_map(df,
                      color_col='n',
                      states_list=[],
                      title="Annual Frequency of Serious Outages by State",
                      figsize=(12, 8),
                      cmap='Reds',
                      reverse_state_abbreviations=reverse_state_abbreviations,
                      gpd_url=gpd_url,
                     ):

    """
    Plots a map of U.S. states colored according to a specified value.
        
    Args:
        df (pd.DataFrame):
            DataFrame containing at least a 'state' column and the column to use for coloring.
        color_col (str, optional):
            Name of the column used to determine state color intensity. Defaults to 'n'.
        states_list (list, optional):
            List of state names to include. If empty, all contiguous U.S. states are shown.
        title (str, optional):
            Title of the plot. Defaults to "Annual Frequency of Serious Outages by State".
        figsize (tuple, optional):
            Size of the figure. Defaults to (12, 8).
        cmap (str, optional):
            Matplotlib colormap to use. Defaults to 'Reds'.
        reverse_state_abbreviations (dict, optional):
            Dictionary mapping full state names to their two-letter abbreviations.
        gpd_url (str):
            URL to load shapefile.
    
    Returns:
        None. Displays a map plot.
    
    Notes:
        - Counties without data are shown in light grey.
        - Puerto Rico, Alaska, and Hawaii are excluded unless explicitly specified in `states_list`.
        - The function requires a shapefile of U.S. counties (e.g., from the U.S. Census Bureau).
    """

    # Load county shapefile
    counties = gpd.read_file(gpd_url)

    # Create color mapping
    color_map = {}
    for i, row in df.iterrows():
        state = row['state']
        color_map[state] = row[color_col]
        if state in reverse_state_abbreviations :
            state_abbr = reverse_state_abbreviations[state]
            color_map[state_abbr] = row[color_col]

    # Create color column
    counties['color'] = counties['STATE_NAME'].map(color_map)

    # Don't plot stats without colors
    #counties = counties[~counties['color'].isna()]
    
    # Don't plot states outside lands
    if states_list :
        counties = counties[counties['STATE_NAME'].isin(states_list)]
    else :
        outside_states = ['Puerto Rico', 'Alaska', 'Hawaii']
        counties = counties[~counties['STATE_NAME'].isin(outside_states)]

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot with colormap
    counties.plot(
        column="color",
        ax=ax,
        cmap=cmap,
        linewidth=0.2,
        edgecolor='black',
        legend=True,
        missing_kwds={'color': '#F6F6F6', 'label': 'No Data'} # Lightgrey
    )

    # Set title
    ax.set_title(title, fontsize=15)
    #ax.axis('off')

    # Show
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------
   
def plot_fips_on_map(df, color_col,
                     title="Annual Frequency of Serious Outages by County",
                     figsize=(12, 8),
                     cmap='Reds',
                     gpd_url=gpd_url,
                    ):
    """
    Plot a colored US counties map using a FIPS-based dataframe.

    Parameters:
    - df: pandas DataFrame with 'fips' and a color column (e.g., 'days_with_serious_outages')
    - color_col: str, column name to color the counties
    - title (str): graph title (optional)
    - figsize: tuple, size of the figure
    - cmap: str, matplotlib colormap name
    - gpd_url: str, url to shapefile for USA map plotting
    """
    # Load county shapefile
    counties = gpd.read_file(gpd_url)

    # Create color mapping
    color_map = {}
    for i, row in df.iterrows():
        color_map[row['fips']] = row[color_col]
    counties['color'] = counties['GEOID'].astype(int).map(color_map)

    # Don't plot fips outside lands
    counties = counties[~counties['color'].isna()]

    # Don't plot states outside lands
    outside_states = ['Puerto Rico', 'Alaska', 'Hawaii']
    counties = counties[~counties['STATE_NAME'].isin(outside_states)]

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot with colormap
    counties.plot(
        column="color",
        ax=ax,
        cmap=cmap,
        linewidth=0.2,
        edgecolor='black',
        legend=True,
        missing_kwds={'color': 'lightgrey', 'label': 'No Data'}
    )

    # Set title
    ax.set_title(title, fontsize=15)
    #ax.axis('off')

    # Show
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------

import cartopy.crs as ccrs
import cartopy.feature as cfeature
def print_points_on_us_map(latitudes, longitudes, title="Location on USA Map", figsize=(11, 7)):
    """
    Plots a set of geographic points on a map of the continental United States.
    
    Parameters:
        - latitudes (array-like): List or array of latitude coordinates.
        - longitudes (array-like): List or array of longitude coordinates.
        - title (str, optional, default = "Location on USA Map"): Title of the output figure.
        - figsize (tuple, optional, default = (11, 7)): Size of the output figure.
    """

    # Set up the USA map
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.LambertConformal())
    ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.Geodetic())  # Continental US
    
    # Add features
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    
    # Plot points
    ax.scatter(longitudes, latitudes, s=1, color='blue', alpha=0.7, transform=ccrs.Geodetic())

    # Show
    plt.title(title)
    plt.show()

# -----------------------------------------------------------------------------------------------------

def plot_correlation_matrix(df, cols, figsize=(5, 5), cmap=custom_cmap):
    """
    Plots a correlation heatmap for a subset of columns in the given DataFrame.
    
    Parameters:
        - df (pd.DataFrame): Input DataFrame containing the data.
        - cols (list): List of column names to compute and plot the correlation matrix.
        - figsize (tuple, optional, default = (5, 5)): Size of the output plot.
        - cmap (str, optional, default = custom_cmap): Color map to use for the heatmap.
    
    Returns:
        None: Displays a heatmap showing the correlation matrix of the specified columns.
    """

    # Compute correlation
    correlation_matrix = df[cols].corr()
    
    # Create a heatmap
    plt.figure(figsize=figsize)
    sns.set_theme(style="white")  # Set a clean theme
    heatmap = sns.heatmap(
                            correlation_matrix,
                            annot=True,  # Annotate the heatmap with correlation coefficients
                            fmt=".2f",   # Format annotations to 2 decimal places
                            cmap=cmap,  # coolwarm
                            cbar=True,   # Show the color bar
                            square=True, # Make the cells square
                            linewidths=0.5,  # Add gridlines
                            #annot_kws={"size": 8, "weight": "bold"}  # Style the annotations
                        )
    
    # Add a title
    plt.title("Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    
    # Show the plot
    plt.show()

# -----------------------------------------------------------------------------------------------------

def complete_dict_fips_with_indirect_neighbours(dict_fips):

    # 2nd pass to have all indirect neighbours
    for main_fips in dict_fips.keys() :
    
        adjacent_fips_1st = dict_fips[main_fips]['neighbours_1st']
        far_neighbours = dict_fips[main_fips]['far_neighbours']
        
        # 2nd-degree neighbours (neighbours of neighbours)
        adjacent_fips_2nd = [dict_fips[fips]['neighbours_1st'] for fips in adjacent_fips_1st if fips in dict_fips] # Get direct neighbours of all 1st-degree neighbours
        adjacent_fips_2nd = [f for ff in adjacent_fips_2nd for f in ff] # Flatten nested lists
        adjacent_fips_2nd += far_neighbours # Add fips closer than 10km to the inital fips
        adjacent_fips_2nd = sorted(set(adjacent_fips_2nd)) # De-duplicate
        adjacent_fips_2nd = [x for x in adjacent_fips_2nd if x not in adjacent_fips_1st+[main_fips]] # exclude 1st-degree neighbours
        dict_fips[main_fips]['neighbours_2nd'] = adjacent_fips_2nd
        
        # 3nd-degree neighbours
        adjacent_fips_3rd = [dict_fips[fips]['neighbours_1st'] for fips in adjacent_fips_2nd if fips in dict_fips] # Get direct neighbours of all 2st-degree neighbours
        adjacent_fips_3rd += [dict_fips[fips]['far_neighbours'] for fips in adjacent_fips_1st if fips in dict_fips] # Add fips closer than 10km of any all 1st-degree neighbours
        adjacent_fips_3rd = [f for ff in adjacent_fips_3rd for f in ff] # Flatten nested lists
        adjacent_fips_3rd = sorted(set(adjacent_fips_3rd)) # De-duplicate
        adjacent_fips_3rd = [x for x in adjacent_fips_3rd if x not in adjacent_fips_1st+adjacent_fips_2nd+[main_fips]] # exclude 1st-degree and 2nd-degree neighbours
        dict_fips[main_fips]['neighbours_3rd'] = adjacent_fips_3rd
    
        # 4nd-degree neighbours
        adjacent_fips_4th = [dict_fips[fips]['neighbours_1st'] for fips in adjacent_fips_3rd if fips in dict_fips] # Get direct neighbours of all 2st-degree neighbours
        adjacent_fips_4th += [dict_fips[fips]['far_neighbours'] for fips in adjacent_fips_2nd if fips in dict_fips] # Add fips closer than 10km of any all 2nd-degree neighbours
        adjacent_fips_4th = [f for ff in adjacent_fips_4th for f in ff] # Flatten nested lists
        adjacent_fips_4th = sorted(set(adjacent_fips_4th)) # De-duplicate
        adjacent_fips_4th = [x for x in adjacent_fips_4th if x not in adjacent_fips_1st+adjacent_fips_2nd+adjacent_fips_3rd+[main_fips]] # exclude 1st-degree and 2nd-degree neighbours
        dict_fips[main_fips]['neighbours_4th'] = adjacent_fips_4th
    
    # Sum land and water areas
    for main_fips in tqdm(dict_fips.keys()) :
        for k in ['neighbours_1st', 'neighbours_2nd', 'neighbours_3rd', 'neighbours_4th'] :
            adjacents_fips = [adj_fips for adj_fips in dict_fips[main_fips][k] if adj_fips in dict_fips]
            dict_fips[main_fips][f'LandArea_{k}'] = np.sum([dict_fips[adj_fips]['LandArea'] for adj_fips in adjacents_fips])
            dict_fips[main_fips][f'WaterArea_{k}'] = np.sum([dict_fips[adj_fips]['WaterArea'] for adj_fips in adjacents_fips])

    # Return
    return dict_fips

# -----------------------------------------------------------------------------------------------------

def read_txt_line_by_line(file_path, start_year=2014, end_year=2023):
    """

    Read the txt file from https://seer.cancer.gov/popdata/download.html
    -> File description : https://seer.cancer.gov/popdata/popdic.html
    
    Efficiently reads the text file line by line, filters rows based on year, and returns a pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the text file.
        start_year (int): Starting year for filtering.
        end_year (int): Ending year for filtering.

    Returns:
        pd.DataFrame: Filtered DataFrame containing structured data.
    """
    data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            year = int(line[0:4])  # Extract and convert year
            if start_year <= year <= end_year:  # Filter early
                row = [
                    year,
                    line[4:6].strip(),  # State
                    int(line[6:11].strip()),  # FIPS code
                    #line[13:14].strip(),  # Race
                    #line[14:15].strip(),  # Origin
                    #line[15:16].strip(),  # Sex
                    int(line[16:18].strip()),  # Age group (01 to 19)
                    int(line[18:26].strip() or 0)  # Population (default to 0 if empty)
                ]
                data.append(row)

        return pd.DataFrame(data, columns=['year', 'state', 'fips', 'Age', 'Population'])

# -----------------------------------------------------------------------------------------------------

def create_envelop(df) :

    """
    Creates an 'envelop' column by applying a two-step rolling window operation on the 'outages' data for each 'fips'.
    
    Parameters:
        - df (pd.DataFrame): Input DataFrame containing 'fips' and 'outages' columns.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'envelop' column.
    """


    # Create envelop
    """
    PROCESS EXPLANATION :
        - Group by fips: Processes outages separately for each fips.
        - First rolling (3): Finds local maxima over 3 rows.
        - Second rolling (3): Smooth these minima using a 3-row average.
        - Fill Missing Values: Replaces any NaN (due to rolling edges) with original outages values.
    """
    df['envelop'] = df.groupby(['fips'])['outages'].rolling(3, center=True).max().rolling(3, center=True).mean().fillna(df['outages']).values

    # Force 0 values where outages and envelop are very low
    df.loc[(df['outages'] < 10) & (df['envelop'] < 10), 'envelop'] = 0

    # Return
    return df

# -----------------------------------------------------------------------------------------------------

from scipy.signal import find_peaks
def peak_detection(df, col, distance=6, prominence=30):
    """
    Detects peaks in a specified column of a DataFrame and adds multiple columns related to peak characteristics.
    
    Parameters:
        - df (pd.DataFrame): Input DataFrame containing the data to analyze.
        - col (str): Name of the column in which peaks should be detected.
        - distance (int, optional, default = 6): Minimum number of samples between peaks.
        - prominence (int, optional, default = 30): Minimum prominence required for a peak to be detected.
    
    Returns:
        pd.DataFrame: DataFrame with additional columns for peak detection, including:
            - 'peak': Flag indicating if a row is a peak.
            - 'peak_prominence': Prominence of the peak.
            - 'start_peak_x_hours_ago': Hours ago the peak started.
            - 'end_peak_in_x_hours': Hours until the peak ends.
    """

    # Reset index to avoid problems
    df = df.reset_index(drop=True)

    # Flag peaks (at least 6 hours between peaks)
    peaks, stats_peaks = find_peaks(df[col].values, distance=distance, prominence=prominence)
    
    # Create a boolean column for peaks
    df['peak'] = 0 # Initialize
    df.loc[peaks, 'peak'] = 1

    #  Create a column to store peak prominence
    df['peak_prominence'] = 0 # Initialize
    df.loc[peaks, 'peak_prominence'] = stats_peaks['prominences']

    # Create a column to store how many hours ago a peak has started
    df['start_peak_x_hours_ago'] = 0 # Initialize
    df.loc[peaks, 'start_peak_x_hours_ago'] = peaks - stats_peaks['left_bases']
    
    # Create a column to store how many hours peak will still last
    df['end_peak_in_x_hours'] = 0 # Initialize
    df.loc[peaks, 'end_peak_in_x_hours'] =  stats_peaks['right_bases'] - peaks

    # Return
    return df

def peak_correction(df, threshold=10):
    """
    Corrects the peaks detected in the DataFrame based on specified thresholds and prominence conditions.
    The idea is to avoid mixing up small peaks and fluctuations on big peaks.
    
    Parameters:
        - df (pd.DataFrame): Input DataFrame with detected peaks and associated columns.
        - threshold (int, optional, default = 10): Minimum value for outages to be considered a peak.
    
    Returns:
        pd.DataFrame: DataFrame with corrected peak-related columns, where small peaks or fluctuations are removed based on conditions:
            - 'peak': Corrected flag for peaks, with small peaks or fluctuations set to 0.
            - 'start_peak_x_hours_ago': Resets values where peaks are below the threshold.
            - 'end_peak_in_x_hours': Resets values where peaks are below the threshold.
    """

    # Apply threshold
    df.loc[(df['outages']<threshold), 'peak'] = 0
    
    # Interpretation : if the prominence is less than 20% of the "floor", it's considered a fluctuation
    mask = (df['peak_prominence'] / df['envelop']) < 0.2
    df.loc[mask, 'peak'] = 0

    # Interpretation : if the prominence is less than x% of a "small floor" (less than thr), it's considered a fluctuation
    for thr, ratio in [[100, 0.5],
                       [200, 0.45],
                       [400, 0.4],
                       [600, 0.35],
                       [1000, 0.3],
                      ]:
        mask = ((df['envelop'] < thr) & ((df['peak_prominence'] / df['envelop']) < ratio))
        df.loc[mask, 'peak'] = 0

    # ---------------------------------------------------------
    
    # A peak has not really started unless it has reached some high values
    df.loc[(df['outages']<threshold), 'start_peak_x_hours_ago'] = 0
    
    # A peak has finished if it has reached some high values
    df.loc[(df['outages']<threshold), 'end_peak_in_x_hours'] = 0

    # Return
    return df

# -----------------------------------------------------------------------------------------------------


def create_gaussian_around_peaks(df, peak_col, col_to_create='outages_outbreak'):
    """
    Creates an asymmetric Gaussian distribution around detected peak events in time series data.
    
    Parameters:
       - df (pd.DataFrame): Input DataFrame containing peak indicators and related metrics.
       - peak_col (str): Name of the column that flags peak events (with value 1).
       - col_to_create (str, optional, default='outages_outbreak'): Name of the output column to store the generated Gaussian values.
    
    Returns:
       pd.DataFrame: Original DataFrame with the new Gaussian distribution column added.
    """
    
    # Initialize column with zeros
    gaussian_col = np.zeros(len(df))

    # Store index of curves left to peaks
    index_left_peaks = set()

    # Iterate over rows flagged as peaks
    for i in df.index[df[peak_col] == 1]:
        # Asymmetric sigmas
        left_radius  = max(3, min(36, df.loc[i, 'start_peak_x_hours_ago'])) # Maximum 36 hours before the peak, min 3
        right_radius = max(3, min(6, df.loc[i, 'end_peak_in_x_hours'])) # Maximum 6 hours after the peak, min 4
        sigma_left   = max(1, min(15, round(df.loc[i, 'start_peak_x_hours_ago'] / 2))) # Maximum 15, minimum 1
        sigma_right  = max(1, min(1, df.loc[i, 'end_peak_in_x_hours'] / 3)) # Maximum 1 to have steep curve after the peak, minimum 1
        # Apply Gaussian weights within the range around the peak
        
        for j in range(max(0, i - left_radius), i):
            index_left_peaks.add(j)
            coef = np.exp(-((j - i) ** 2) / (2 * sigma_left ** 2))
            gaussian_col[j] = max(coef, gaussian_col[j])
        # Apply Gaussian weights within the range around the peak
        for j in range(i, min(len(df), i + right_radius + 1)):
            coef = np.exp(-((j - i) ** 2) / (2 * sigma_right ** 2))
            gaussian_col[j] = max(coef, gaussian_col[j])

    # Multiply our envelop by gaussian weights
    df[col_to_create] = df['envelop'] * gaussian_col

    # At the left of peaks, we take the max between our gaussian and the real outages number (but only where our gaussian has non null values)
    mask = (df.index.isin(index_left_peaks)) & (df[col_to_create] > 0)
    df.loc[mask, col_to_create] = df.loc[mask, ['outages', col_to_create]].max(axis=1)

    # Fill NaN
    df[col_to_create] = df[col_to_create].fillna(0)
    
    # Return
    return df

# -----------------------------------------------------------------------------------------------------

def find_adjacent_fips(counties_gdf, main_fips):
    """
    Finds all adjacent FIPS codes to a given main FIPS code based on shared boundaries.

    Parameters:
        - counties_gdf (GeoDataFrame): GeoDataFrame containing polygons with 'GEOID' (FIPS code) and geometries.
       -  main_fips (str): The FIPS code to find neighbors for.

    Returns:
        list: A list of adjacent FIPS codes.
    """
    # Filter the GeoDataFrame to isolate the main FIPS
    main_fips_polygon = counties_gdf.loc[counties_gdf['GEOID'].astype(int) == int(main_fips)]

    if main_fips_polygon.empty:
        #print(f"FIPS code {main_fips} not found in GeoDataFrame.")
        return [], []

    # Find all counties that touch the main FIPS polygon
    neighbors = counties_gdf[counties_gdf.geometry.touches(main_fips_polygon.geometry.values[0])]

    # Extract the FIPS codes of the neighbors
    adjacent_fips = [int(x) for x in neighbors['GEOID'] if int(x)!=main_fips]

    # ==================================================================
    # FIND NON ADJACENT NEIGHBOURS, but separated by a very short distance (lake, rivers ...)
    # Ensure a projected CRS for accurate distance in meters
    counties_projected = counties_gdf.to_crs(epsg=5070)  # US National Atlas Equal Area
    main_geom_projected = counties_projected[counties_projected['GEOID'].astype(int) == int(main_fips)].iloc[0].geometry

    # Compute distance
    counties_projected['distance'] = counties_projected.geometry.distance(main_geom_projected)

    # Find neighbors within threshold
    close_neighbours = counties_projected[counties_projected['distance'].between(0, 10000)] # distance < 10,000 meters
    close_neighbours = [int(x) for x in close_neighbours['GEOID'] if int(x)!=main_fips]

    # Find 2nd degree neighbors
    far_neighbours = counties_projected[counties_projected['distance'].between(10001, 20000)]  # distance 10,001 to 20,000 meters
    far_neighbours = [int(x) for x in far_neighbours['GEOID']]

    # Deduplicate
    adjacent_fips = list(set(adjacent_fips) | set(close_neighbours))

    # Return
    return adjacent_fips, far_neighbours


# -----------------------------------------------------------------------------------------------------

def plot_fips_and_neighbors_on_map(df,
                                   fips,
                                   dict_fips,
                                   axe=None,
                                   figsize=(10, 6),
                                   legend='best',
                                   gpd_url=gpd_url,
                                  ):
    """
    Plots a FIPS code and its neighbors on a map with custom coloring.

    Parameters:
        - df (pd.DataFrame): DataFrame containing 'state' and 'fips_code'.
        - fips (str): The main FIPS code to highlight.
        - dict_fips (dict of lists): Dictionary mapping FIPS codes to their neighboring regions at different degrees of separation.
                                     Must include keys 'neighbours_1st', 'neighbours_2nd', 'neighbours_3rd', 'neighbours_4th'.
        - gpd_url (str) : url to shapefile for USA map plotting.
        
    """

    # Load county shape file
    # Source : https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_20m.zip
    counties = gpd.read_file(gpd_url)

    # Filter by state
    list_of_fips = [fips] +\
                   dict_fips[fips]['neighbours_1st'] +\
                   dict_fips[fips]['neighbours_2nd'] +\
                   dict_fips[fips]['neighbours_3rd'] +\
                   dict_fips[fips]['neighbours_4th']
    # > Method n°1 : if 'states' is in df
    if 'state' in df :
        states = df.loc[df['fips'].isin(list_of_fips), 'state'].unique()
        counties = counties[counties['STUSPS'].isin(states)] # STUSPS or STATE_NAME
    # > Method n°2 : if 'states' is not in df
    else :
        states_codes = list(set([str(x).zfill(5)[:2] for x in list_of_fips]))
        mask = counties['GEOID'].astype(str).apply(lambda x : x.zfill(5)[:2]).isin(states_codes)
        counties = counties[mask] # STUSPS or STATE_NAME

    # Define a custom column for identifying FIPS and neighbors
    counties['highlight'] = counties['GEOID'].astype(int).apply(lambda x: 'main' if x == fips else 'other')
    for x in ['neighbours_1st', 'neighbours_2nd', 'neighbours_3rd', 'neighbours_4th'] :
        counties.loc[counties['GEOID'].astype(int).isin(dict_fips[fips][x]), 'highlight'] = x
    
    # Create a custom colormap
    cmap = {'main': find_color_neighbours('main'),
            'neighbours_1st': find_color_neighbours('neighbours_1st'),
            'neighbours_2nd': find_color_neighbours('neighbours_2nd'),
            'neighbours_3rd': find_color_neighbours('neighbours_3rd'),
            'neighbours_4th': find_color_neighbours('neighbours_4th'),
            'other': '#E0E0E0'}
    counties['color'] = counties['highlight'].map(cmap)

    # Create plot figure
    if axe is None :
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else :
        ax = axe

    # Plot the map
    counties.plot(ax=ax, color=counties['color'], linewidth=0.8, edgecolor='black')
    # Add title and annotations
    title = f"FIPS {fips} Neighbourhood"
    ax.set_title(title, fontsize=14)
    ax.axis('off')  # Turn off the axis

    # Add a custom legend
    legend_elements = [
        Patch(facecolor=cmap['main'], edgecolor='black', label=f'Main FIPS ({fips})'),
        Patch(facecolor=cmap['neighbours_1st'], edgecolor='black', label='1st Neighbours'),
        Patch(facecolor=cmap['neighbours_2nd'], edgecolor='black', label='2nd Neighbours'),
        Patch(facecolor=cmap['neighbours_3rd'], edgecolor='black', label='3rd Neighbours'),
        Patch(facecolor=cmap['neighbours_4th'], edgecolor='black', label='4th Neighbours'),
    ]

    # Place legend to avoid over-writing on the graph
    if legend == 'right' :
        ax.legend(handles=legend_elements, loc='center', fontsize=10, bbox_to_anchor=(1.14, 0.5))
    elif legend == 'left' :
        ax.legend(handles=legend_elements, loc='center', fontsize=10, bbox_to_anchor=(-0.14, 0.5))
    else :
        ax.legend(handles=legend_elements, loc='best', fontsize=10)

    # Show
    if axe is None :
        plt.show()

# -----------------------------------------------------------------------------------------------------


import matplotlib.dates as mdates
def plot_fips_and_neighbors_outages(df, fips, dict_fips, start_date, end_date, k='outages_outbreak', axe=None, figsize=(8, 5)):
    """
    Plots a column "col" values for a given FIPS and its neighboring FIPS, grouped by different proximity levels (1st, 2nd, 3rd, 4th neighbors), within a specified date range.
    
    Parameters:
        - df (pd.DataFrame): Input DataFrame containing outage data with columns for 'fips', 'date', 'hour', and column "k".
        - fips (int): FIPS code for the main location to plot.
        - dict_fips (dict): Dictionary containing neighboring FIPS information for each FIPS, with keys 'neighbours_1st', 'neighbours_2nd', 'neighbours_3rd', and 'neighbours_4th'.
        - start_date (str or None): The start date for the date range to filter the data (format: 'YYYY-MM-DD').
        - end_date (str or None): The end date for the date range to filter the data (format: 'YYYY-MM-DD').
        - k (str): column of df to plot.
        - axe (matplotlib.axes.Axes, optional): Matplotlib axes object to plot on; if None, a new figure is created.
        - figsize (tuple, optional, default=(8, 5)): Size of the plot figure.
    
    Returns:
        None: Displays the plot with the normalized outage values for the main FIPS and its neighboring regions.
    """
    # Filter DataFrame for the specified FIPS and date range
    mask_date = (~df['fips'].isna())
    if start_date is not None : mask_date = mask_date & (df['date'] >= start_date)
    if end_date is not None   : mask_date = mask_date & (df['date'] <= end_date)   


    # Plot main fips
    mask = (df['fips'] == fips) & mask_date
    df_tmp = df[mask].reset_index(drop=True)
    df_tmp['datetime'] = pd.to_datetime(df_tmp['date']) + pd.to_timedelta(df_tmp['hour'], unit='h')
    x = df_tmp['datetime'].values
    y = df_tmp[k].values

    # Early stopping
    if len(df_tmp) == 0 :
        print("Nothing to plot !")
        return

    # 1st neighbours
    mask = (df['fips'].isin(dict_fips[fips]['neighbours_1st'])) & mask_date
    df_tmp = df[mask].reset_index(drop=True)
    df_tmp = df_tmp.groupby(['date', 'hour'])[k].sum().to_frame().reset_index()
    y1 = df_tmp[k].values

    # 2nd neighbours
    mask = (df['fips'].isin(dict_fips[fips]['neighbours_2nd'])) & mask_date
    df_tmp = df[mask].reset_index(drop=True)
    df_tmp = df_tmp.groupby(['date', 'hour'])[k].sum().to_frame().reset_index()
    y2 = df_tmp[k].values

    # 3rd neighbours
    mask = (df['fips'].isin(dict_fips[fips]['neighbours_3rd'])) & mask_date
    df_tmp = df[mask].reset_index(drop=True)
    df_tmp = df_tmp.groupby(['date', 'hour'])[k].sum().to_frame().reset_index()
    y3 = df_tmp[k].values

    # 4th neighbours
    mask = (df['fips'].isin(dict_fips[fips]['neighbours_4th'])) & mask_date
    df_tmp = df[mask].reset_index(drop=True)
    df_tmp = df_tmp.groupby(['date', 'hour'])[k].sum().to_frame().reset_index()
    y4 = df_tmp[k].values

    # Create plot figure
    if axe is None :
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else :
        ax = axe

    # Plot lines
    ax.plot(x, y/y.max(), label=f'Fips n°{fips}', linewidth=2.5, color=find_color_neighbours('main_fips'),)
    if len(y1)>0: ax.plot(x, y1/y1.max(), label='1st neighbours', linewidth=2.5, color=find_color_neighbours('neighbours_1st'),)
    if len(y2)>0: ax.plot(x, y2/y2.max(), label='2nd neighbours', linewidth=2.5, color=find_color_neighbours('neighbours_2nd'),)
    if len(y3)>0: ax.plot(x, y3/y3.max(), label='3rd neighbours', linewidth=2.5, color=find_color_neighbours('neighbours_3rd'),)
    if len(y4)>0: ax.plot(x, y4/y4.max(), label='4th neighbours', linewidth=2.5, color=find_color_neighbours('neighbours_4th'),)

    # Fill zones under curves
    ax.fill_between(x, y/y.max(), color='yellow', alpha=0.1, linewidth=0)
    if len(y1)>0: ax.fill_between(x, y1/y1.max(), alpha=0.1, linewidth=0, color=find_color_neighbours('neighbours_1st'),)
    if len(y2)>0: ax.fill_between(x, y2/y2.max(), alpha=0.1, linewidth=0, color=find_color_neighbours('neighbours_2nd'),)
    if len(y3)>0: ax.fill_between(x, y3/y3.max(), alpha=0.1, linewidth=0, color=find_color_neighbours('neighbours_3rd'),)
    if len(y4)>0: ax.fill_between(x, y4/y4.max(), alpha=0.1, linewidth=0, color=find_color_neighbours('neighbours_4th'),)

    # Graph parameters
    ax.legend(loc='best')
    ax.set_title(f'Outage Outbreaks for fips n°{fips} and its neighbourhood', fontsize=14)
    ax.set_ylabel('Normalized value of "outage outbreaks"')
    # Format x-axis to display only the date of the timestamps
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_xticks(x[::24])  # Set x-ticks to unique dates (every 24 hours)
    
    # Show
    if axe is None :
        plt.show()

# -----------------------------------------------------------------------------------------------------

def convert_damage(damage):
    """
    Converts damage_property values like '10.00K', '10.00M' or '1B' into full integers.
    
    Parameters:
        - damage (str): The damage value as a string (e.g., '10.00K', '10.00M').
    
    Returns:
        int: The full integer representation of the damage value.
    """
    if damage.endswith('K'):
        return int(float(damage[:-1]) * 1e3)  # Multiply by 1,000 for 'K'
    elif damage.endswith('M'):
        return int(float(damage[:-1]) * 1e6)  # Multiply by 1,000,000 for 'M'
    elif damage.endswith('B'):
        return int(float(damage[:-1]) * 1e9)  # Multiply by 1,000,000 for 'M'
    elif damage.endswith('n'): # nan
        return 0
    else:
        return int(float(damage))  # No suffix means it's already a number

# -----------------------------------------------------------------------------------------------------

def plot_bars(stats, col, title=None, figsize=(14, 3)):
    """
    Plots a bar chart for aggregated statistics, highlighting the proportion of each category.
    
    Parameters:
        - stats (pd.DataFrame): DataFrame containing aggregated statistics (resulting from a groupby operation).
        - col (str): Name of the column in the 'stats' DataFrame to plot.
        - title (str, optional): Title of the plot.
        - figsize (tuple, optional, default=(14, 3)): Size of the figure (width, height).
    """

    # Sort
    stats = stats.sort_values(by = [col], ascending=False)
    # Data to plot
    values = stats[col].values
    proportions = values / values.sum()
    # Red colormap
    colors = plt.cm.Reds(proportions/max(proportions))
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True) # Put grid behind bars
    # Create the bar plot
    plt.bar(stats.iloc[:, 0], values, color=colors)
    # Rotate the x-axis labels by 90 degrees
    plt.xticks(rotation=90, fontsize=10)
    # Add proportions as text above the bars
    vertical_space = 0.03*max(values)
    for i, (count, prop) in enumerate(zip(values, proportions)):
        plt.text(i, count+vertical_space, f'{100*prop:.1f}%', ha='center', fontsize=10, color='grey', rotation=90)
    # Add title
    if title is None :
        plt.title(k)
    else :
        plt.title(title)
    # Show
    plt.show()

# -----------------------------------------------------------------------------------------------------

def plot_stats_about_severity_groups(df_eaglei) :
    """
    Generates and displays dual bar plots to visualize the total number of events and average property damage by event severity groups.
    
    Parameters:
        - df_eaglei (pd.DataFrame): Input DataFrame containing event data with columns 'Event_Severity', 'EPISODE_ID', and 'DAMAGE_PROPERTY'.
    
    Returns:
        None: Displays two bar plots side by side.
    """


    # Compute stats
    stats = df_eaglei.groupby(['Event_Severity']).agg({'EPISODE_ID' : ['nunique'],
                                                       'DAMAGE_PROPERTY' : ['mean'],
                                                      }).reset_index()
    # Flatten columns
    stats.columns = ['_'.join(i).rstrip('_') for i in stats.columns.values]

    # Define labels and colors for the severity levels
    labels = ['0 (Light)', '1 (Moderate)', '2 (Severe)']
    colors = ['#42D674', '#CCAA7A', '#c54f4d']
    
    # Data
    counts  = stats['EPISODE_ID_nunique']
    damages = stats['DAMAGE_PROPERTY_mean']
    
    # Create a figure and axis for the dual-axis plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)) # 1 rows, 2 columns
    
    # Create the bar plot n°1+
    axes[0].set_title("Total Events by Severity Group")
    axes[0].bar(labels, counts, color=colors)
    axes[0].set_xlabel("Severity", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    
    
    # Create the bar plot n°2
    axes[1].set_title("Average Damage by Severity Group")
    axes[1].bar(labels, damages, color=colors)
    axes[1].set_xlabel("Severity", fontsize=12)
    axes[1].set_ylabel("Average Property Damage (in $)", fontsize=12)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------

def plot_stats_about_episodes(df_eaglei) :
    """
    Generates and displays a set of four plots to visualize various statistics about episodes, including total property damage, episode duration, number of events, and number of event types.
    
    Parameters:
        - df_eaglei (pd.DataFrame): Input DataFrame containing episode data with columns 'EPISODE_ID', 'EVENT_ID', 'EVENT_TYPE', 'BEGIN_DATE_TIME', 'END_DATE_TIME', and 'DAMAGE_PROPERTY'.
    
    Returns:
        None: Displays four plots in a 2x2 grid.
    """


    # Dict of aggregations
    dict_agg = {'EVENT_ID' : ['nunique'],
                'EVENT_TYPE' : ['nunique'],
                'BEGIN_DATE_TIME' : ['min'],
                'END_DATE_TIME' : ['max'],
                'DAMAGE_PROPERTY' : ['sum'],
               }
    
    # Groupby
    df_episodes = df_eaglei.groupby(['EPISODE_ID']).agg(dict_agg).reset_index()
    
    # Flatten column names
    df_episodes.columns = ['_'.join(i).rstrip('_') for i in df_episodes.columns.values]
    
    # Duration (in hours)
    df_episodes.insert(6, "duration_hours", round((df_episodes['END_DATE_TIME_max'] - df_episodes['BEGIN_DATE_TIME_min']).dt.total_seconds() / 3600, 1))
    
    # Create a figure and axis for the dual-axis plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 7)) # 2 rows, 2 columns
    
    # Total Property Damage by Episode
    axes[0, 0].set_title("Total Property Damage by Episode")
    sns.kdeplot(df_episodes['DAMAGE_PROPERTY_sum'].values, ax=axes[0, 0], clip=(0, 1e8), shade=True, color='#FF746C')
    axes[0, 0].set_xlabel("Damage (in $)", fontsize=12)
    axes[0, 0].set_ylabel("Frequency", fontsize=12)
    axes[0, 0].grid(axis='x', linestyle='--', alpha=0.3)
    
    
    # Episodes' durations
    axes[0, 1].set_title("Episodes Duration")
    sns.kdeplot(df_episodes['duration_hours'].values, ax=axes[0, 1], clip=(0, None), shade=True, color='#FFC067')
    axes[0, 1].set_xlabel("Duration (in hours)", fontsize=12)
    axes[0, 1].set_ylabel("Frequency", fontsize=12)
    axes[0, 1].grid(axis='x', linestyle='--', alpha=0.3)
    
    # Number of events by episodes
    axes[1, 0].set_title("Number of Events by Episode")
    sns.kdeplot(df_episodes['EVENT_ID_nunique'].values, ax=axes[1, 0], clip=(0, None), shade=True, color='#1f77b4', alpha=0.8)
    axes[1, 0].set_xlabel("Count of Event", fontsize=12)
    axes[1, 0].set_ylabel("Frequency", fontsize=12)
    axes[1, 0].grid(axis='x', linestyle='--', alpha=0.3)
    
    # Number of event types by episodes
    axes[1, 1].set_title("Number of Event Types by Episode")
    axes[1, 1].hist(df_episodes['EVENT_TYPE_nunique'].values, bins=df_episodes['EVENT_TYPE_nunique'].nunique())
    axes[1, 1].set_xlabel("Count of Event Types", fontsize=12)
    axes[1, 1].set_ylabel("Frequency", fontsize=12)
    axes[1, 1].grid(axis='x', linestyle='--', alpha=0.3)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------

# Dictionnary of labels corresponding to columns
dict_labels = {'Severe_events_count' : 'Severe weather Events',
        		'Count_Other_moderate' : 'Moderate weather Events',
        		'Count_Other_light' : 'Light weather Events',
                'Severe_events_count_sum_among_1st_neighbours' : 'Severe weather - 1st Neighbours',
                'Severe_events_count_sum_among_2nd_neighbours' : 'Severe weather - 2nd Neighbours',
                'Severe_events_count_sum_among_3rd_neighbours' : 'Severe weather - 3rd Neighbours',
                'Severe_events_count_sum_among_4th_neighbours' : 'Severe weather - 4th Neighbours',
                'Severe_events_count_sum_among_ALL_neighbours' : 'Severe weather - All Neighbours',

                'WORDS_wind_and_storm' : 'NARRATIVE Wind & Storms',
                'WORDS_downed_trees' : 'NARRATIVE Downed Trees',
                'WORDS_heavy_damage' : 'NARRATIVE Heavy Damages',
                'WORDS_wires' : 'NARRATIVE Wires',

                'WORDS_wind_and_storm_sum_among_ALL_neighbours' : 'NARRATIVE Wind & Storms - All Neighbours',
                'WORDS_downed_trees_sum_among_ALL_neighbours' : 'NARRATIVE Downed Trees - All Neighbours',
                'WORDS_heavy_damage_sum_among_ALL_neighbours' : 'NARRATIVE Heavy Damages - All Neighbours',
                'WORDS_wires_among_sum_ALL_neighbours' : 'NARRATIVE Wires - All Neighbours',

        		'max_state_fips_with_more_than_50_outages_over_last_6hours' : 'State fips with >50 outages (over last 6 hours)',
                'max_Proportion_of_fips_within_300km_with_more_than_50_outages_over_last_24hours' : 'Fips within 300km with >50 outages (over last 24 hours)',
                'projected_impact_in_6hours' : 'Projected Impact in 6 hours',
               }


def plot_peaks(df,
              fips,
              cols_to_plot=['outages', 'outages_outbreak'],
              cols_timeline=[],
              start_date=None,
              end_date=None,
              plot_timeline_with_intensity = True,
              dict_labels=dict_labels,
              figsize=(14, 5),
              ):
    """
    Plots hourly outages and events timeline for a specified FIPS code.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing outage and event data.
    - fips (str or int): FIPS code to filter the data.
    - cols_to_plot (list, optional, default=['outages', 'outages_outbreak']): List of columns to plot on the main graph.
    - cols_timeline (list, optional, default=[]): List of columns representing weather events to plot on the secondary timeline.
    - start_date (str or datetime, optional): Start date for filtering the data.
    - end_date (str or datetime, optional): End date for filtering the data.
    - plot_timeline_with_intensity (bool, optional, default=True): If True, plots the weather events timeline with intensity.
    - dict_labels (dict, optional): Dictionary mapping column names to custom labels for the timeline.
    - figsize (tuple, optional, default=(14, 5)): Size of the figure (width, height).

    Returns:
        None: Displays the plot with outages and weather events timeline.
    """


    # Filter DataFrame for the specified FIPS and create a filter for the date range
    mask = (df['fips'] == fips)
    if start_date is not None : mask = mask & (df['date'] >= start_date)
    if end_date is not None   : mask = mask & (df['date'] <= end_date)   

    # Create a temporary df
    df_tmp = df[mask].reset_index(drop=True)

    # Early stopping
    if len(df_tmp) == 0 :
        print("Nothing to plot !")
        return

    # Get state
    if 'state' in df_tmp.columns :
        state = df_tmp['state'].values[0]
    else :
        state = 'Unknown state'

    # Create datetime column if not exists
    if 'datetime' not in df_tmp:
        if 'minute' in df_tmp :
            df_tmp['datetime'] = pd.to_datetime(df_tmp['date']) + pd.to_timedelta(df_tmp['hour'], unit='h') + pd.to_timedelta(df_tmp['minute'], unit='m')
        else :
            df_tmp['datetime'] = pd.to_datetime(df_tmp['date']) + pd.to_timedelta(df_tmp['hour'], unit='h')
    
    # --------------------------------------------------------------------------
    # Weather Events columns
    
    # Select columns in df
    cols_timeline = [k for k in cols_timeline if k in df.columns]

    # Decide subplot layout: 2 subplots if weather events exist (to plot a timeline), otherwise 1 subplot
    if cols_timeline:
        ax1_height = figsize[1] # Fixed height for ax1
        ax2_height = np.clip(len(cols_timeline)*0.5, 0, 3) # Different sizes according to how many timelines are plot

        figsize = (figsize[0], ax1_height+ax2_height)
        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [ax1_height*2, round(ax2_height*2)]})
        ax1, ax2 = axs
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None  # No weather timeline
    
    # --------------------------------------------------------------------------
    # Create Main Plot
    
    # --- Main Plot (Outages) ---
    ax1.set_title(f'Hourly Outages (FIPS {fips})')

    # Plot on ax1
    cols_to_plot_on_ax1 = [x for x in cols_to_plot if x not in cols_timeline]
    for x in cols_to_plot_on_ax1:
        
        # Plot envelop (discrete dashed line, grey)
        if x=='envelop' :
            ax1.plot(df_tmp['datetime'], df_tmp['envelop'], color='#ff7f0e', alpha=0.8, linestyle='--', label='Envelop')
    
        # Plot outbreaks (filled area under the curve, pastel blue)
        elif x=='outages_outbreak' :
            ax1.fill_between(df_tmp['datetime'], df_tmp['outages_outbreak'], color='#1f77b4', alpha=0.25, label='Detected Outage Outbreak')
    
        # Scatter plot peaks (red dots)
        elif x=='peak' :
            peaks = df_tmp[df_tmp['peak'] == 1]
            ax1.scatter(peaks['datetime'], peaks['outages'], color='red', label='Peaks', zorder=5)
            
        else:
            ax1.plot(df_tmp['datetime'], df_tmp[x], alpha=0.8, label=x)

    # Ax1 parameters
    ax1.set_ylabel("Outages")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # --------------------------------------------------------------------------
    # Create Secondary Plot : Weather Events Timeline

    if ax2 is not None :
    
        ax2.set_title('Timeline')

        labels, yticks = [], []

        i = -1 # Bar position
        for k in cols_timeline:
            i += 1
            # Stack events downwards
            y_pos = -i
            yticks.append(y_pos)
            label = dict_labels[k] if k in dict_labels else k
            labels.append(label)
            # Simple plot without intensity
            ax2.fill_between(df_tmp['datetime'], y_pos - 0.3, y_pos + 0.3,
                             where=df_tmp[k] > 0,
                             color=find_color_neighbours(k),
                             alpha=0.65-0.35*plot_timeline_with_intensity,
                             label=label)
            # Add intensity nuances to timeline
            if plot_timeline_with_intensity:
                # Get the weather event values and normalize them
                values = df_tmp[k].values
                maxmin_diff = np.nanmax(values) - np.nanmin(values)
                if maxmin_diff > 0:
                    norm = (values - np.nanmin(values)) / maxmin_diff  # Normalize values to 0-1 range
                else:
                    norm = values
                # Find color for the current column
                color = find_color_neighbours(k)
                # Iterate over the rows to create segments based on intensity
                for j in range(1, len(values)):
                    if values[j-1] > 0:  # Only plot for weather event periods (val > 0)
                        end_index = j
                        if values[end_index] > 0:
                            end_index += 1
                        alpha = max(0, 0.6*norm[j-1])
                        ax2.fill_between(df_tmp['datetime'].iloc[j-1:end_index], y_pos - 0.3, y_pos + 0.3,
                                             color=color, alpha=np.clip(alpha, 0, 0.8), edgecolor='none')

        # Ax2 parameters
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(labels, fontsize=10)
        ax2.grid(alpha=0.3)

        # Format x-axis for better readability
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        #plt.xticks(rotation=45)

    # --------------------------------------------------------------------------
    # Plot
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------

def add_every_hour_between_periods(df_tmp) :

    """
    Create a dataframe with every hour between the 'BEGIN_DATE_TIME' and 'END_DATE_TIME' periods
    found in input dataframe (with a row for each hour of each period).
    
    Parameters:
        - df_tmp (pd.DataFrame): DataFrame containing the columns 'BEGIN_DATE_TIME', 'END_DATE_TIME'.
    
    Returns: pd.DataFrame: A DataFrame with additional rows for each hour between the 'BEGIN_DATE_TIME' and 'END_DATE_TIME' for every event.
    """

    # Round hours
    df_tmp['BEGIN_DATE_TIME'] = df_tmp['BEGIN_DATE_TIME'].apply(lambda x : x.floor(freq='H'))
    df_tmp['END_DATE_TIME'] = df_tmp['END_DATE_TIME'].apply(lambda x : x.ceil(freq='H'))

    # Get all periods
    df_hour_pairs = df_tmp[['BEGIN_DATE_TIME', 'END_DATE_TIME']].drop_duplicates().reset_index(drop=True)
    
    # Round hours
    starts = df_hour_pairs['BEGIN_DATE_TIME'].values
    ends   = df_hour_pairs['END_DATE_TIME'].values
    
    # Build all ranges as one large array
    all_ranges = [
                   np.arange(start, end + np.timedelta64(1, 'h'), np.timedelta64(1, 'h'))
                   for start, end in zip(starts, ends)
                 ]
    
    # Flatten and delete duplicates
    set_of_date_ranges = set(np.concatenate(all_ranges))

    # Create a df with all hours
    df_all_hours = pd.DataFrame({'date': sorted(set_of_date_ranges)}).reset_index(drop=True)

    # Merge with df_tmp
    df_all_hours = pd.merge(df_all_hours, df_tmp, how='cross')

    # Filter valid rows & sort values
    mask = (df_all_hours['date'].between(df_all_hours['BEGIN_DATE_TIME'], df_all_hours['END_DATE_TIME']))
    df_all_hours = df_all_hours[mask].sort_values(by = ['date', 'BEGIN_DATE_TIME', 'EVENT_ID']).reset_index(drop=True)

    # Return
    return df_all_hours

# -----------------------------------------------------------------------------------------------------

def add_number_of_fips_with_high_outages_in_radius(df,
                                                   df_latlong,
                                                   thr_outages=50,
                                                   radius=[300, 600],
                                                  ):

    """
    Adds features related to the proportion of nearby FIPS with high outages within specified radius.
    
    For each FIPS, the function calculates the proportion of neighboring FIPS within given distances 
    having outages above a threshold, and derives rolling maximum features over 24 hours.
    
    Args:
        df (pd.DataFrame):
            Main DataFrame containing at least 'fips', 'outages', 'date', and 'hour' columns.
        df_latlong (pd.DataFrame):
            DataFrame containing 'fips', 'lat', and 'lon' columns with FIPS coordinates.
        thr_outages (int, optional):
            Outage threshold to flag high-outage FIPS. Defaults to 50.
        radius (list, optional):
            List of distance thresholds (in kilometers) to consider. Defaults to [300, 600].
    
    Returns:
        pd.DataFrame:
            Updated DataFrame with new proximity-based features.
    """


    # Number of fips with outages > thr
    df[f'flag_more_than_{thr_outages}_outages'] = (df['outages'] >= 50).astype(int)

    # Coordinates of fips
    coords = df_latlong[['lat', 'lon']].values

    # Precompute full haversine distance matrix
    distance_matrix = haversine_vector(coords, coords, Unit.KILOMETERS, comb=True)
    
    # Set fips as index for faster process
    df.index = df['fips'].astype(int).values

    # New columns that we will creats
    new_cols = set()
    
    # Iterate over radius
    for dist_thr in radius:
        
        # Create a distance mask
        neighbors_mask = (distance_matrix <= dist_thr)

        # Iterate over fips
        fips_list = df_latlong['fips'].unique()
        for i, fips in tqdm(enumerate(fips_list), position=0, leave=True, total=len(fips_list)):

            # Continue if fips is not in df
            if fips not in df['fips'].unique():
                continue
            
            # Get nearby fips within dist_thr
            nearby_fips = df_latlong.loc[neighbors_mask[i], 'fips'].unique()
                
            # Create data
            col_name = f"Proportion_of_fips_within_{dist_thr}km_with_more_than_{thr_outages}_outages"
            df.loc[fips, col_name] = df.loc[nearby_fips].groupby(['date', 'hour'])[f'flag_more_than_{thr_outages}_outages'].mean().values

            # Save column name
            new_cols.add(col_name)
    
    # Compute rolling features
    for k in sorted(new_cols):
        for n_hours in [24]:
            df[f'max_{k}_over_last_{n_hours}hours'] = df.groupby('fips')[k].rolling(n_hours).max().fillna(0).values
    
    # Reset index
    df = df.reset_index(drop=True)

    # Drop columns
    #df = df.drop(columns = [f'flag_more_than_{thr_outages}_outages'])

    # Return
    return df

# -----------------------------------------------------------------------------------------------------

def add_data_from_states(df):
    """
    Adds aggregated and rolling features from state-level data to the input DataFrame.
    
    Parameters:
        - df (pd.DataFrame): DataFrame containing state-level data with columns 'state', 'date', 'hour', and various event-related features.
    
    Returns:
        - pd.DataFrame: DataFrame with added state-level aggregated and rolling features.
    """


    # Groupby cols
    groupby_cols = ['state', 'date', 'hour']

    # Features to aggregate by neighbours
    dict_agg = {'Severe_events_count': ['sum'],
                'WORDS_wind_and_storm' : ['sum'],
                'WORDS_downed_trees' : ['sum'],     
                'WORDS_heavy_damage' : ['sum'],
                'WORDS_wires' : ['sum'],
                'flag_more_than_50_outages' : ['sum'],
                'Count_Other_light' : ['sum'],
                'Count_Other_moderate' : ['sum'],
                'Count_Other_severe' : ['sum'],
                'Count_Flood' : ['sum'],
                'Count_Storm' : ['sum'],
                'Count_Wind' : ['sum'],
                'Count_Fire' : ['sum'],
               }

    # Compute data
    df_tmp = df.groupby(groupby_cols).agg(dict_agg).reset_index()
    
    # Flatten column names
    df_tmp.columns = groupby_cols + ['_'.join(i).rstrip('_') + '_by_state' for i in df_tmp.columns[len(groupby_cols):]]

    # Sort_values
    df_tmp = df_tmp.sort_values(by = groupby_cols).reset_index(drop=True)

    # Rename
    df_tmp = df_tmp.rename(columns = {'flag_more_than_50_outages_sum_by_state' : 'state_fips_with_more_than_50_outages'})

    # Compute rolling features
    k = 'state_fips_with_more_than_50_outages'
    for n_hours in [6, 24]:
        df_tmp[f'max_{k}_over_last_{n_hours}hours'] = df_tmp.groupby('state')[k].rolling(n_hours).max().fillna(0).astype(np.int32).values
        df_tmp[f'proportion_{k}_over_last_{n_hours}hours'] = df_tmp.groupby('state')[k].rolling(n_hours).mean().fillna(0).values

    cols = ['Severe_events_count_sum_by_state',
            'WORDS_wind_and_storm_sum_by_state',
            'WORDS_downed_trees_sum_by_state',
            'WORDS_heavy_damage_sum_by_state',
            'WORDS_wires_sum_by_state',
            'state_fips_with_more_than_50_outages']
    for k in cols:
        for n_hours in [24]:
            df_tmp[f'SUM_BY_STATE_{k}_last_{n_hours}hours'] = df_tmp.groupby('state')[k].rolling(n_hours).sum().fillna(0).astype(np.int32).values
        # Delete original columns (value for only 1 hour), because we only keep rolling feats
        df_tmp = df_tmp.drop(columns = [k])

    # Merge to add data on df
    df = pd.merge(df, df_tmp, how='left', on=groupby_cols)
                
    # Return
    return df

    
# -----------------------------------------------------------------------------------------------------

def add_data_from_neighbours(df, dict_fips):
    """
    Adds aggregated features from neighboring FIPS codes to the input DataFrame.

    Parameters:
        - df (pd.DataFrame): Input DataFrame containing FIPS-level data with columns 'fips', 'date', 'hour', and various event-related features.
        - dict_fips (dict): Dictionary mapping FIPS codes to their neighboring FIPS codes, categorized by range (e.g., '1st', '2nd', '3rd', '4th').

    Returns:
        pd.DataFrame: DataFrame with added neighbor-level aggregated features.
    """


    # Groupby cols
    cols_date = ['date', 'hour']

    # Put fips as index to select row faster
    df.index = df['fips'].values

    # Set of fips code in df (for faster process)
    fips_set = set(df['fips'])

    # Features to aggregate by neighbours
    dict_agg = {'outages' : ['sum'],
                'outages_outbreak' : ['sum'],
                'Severe_events_count' : ['sum'],
                'Temperature_1day_before' : ['std'],
                'MAGNITUDE' : ['sum'],
                'Population' : ['sum'],
                'WORDS_wind_and_storm' : ['sum'],
                'WORDS_downed_trees' : ['sum'],     
                'WORDS_heavy_damage' : ['sum'],
                'WORDS_wires' : ['sum'],
                'WORDS_water_and_flood' : ['sum'],
                'WORDS_heat_and_fire' : ['sum'],   

                'Count_Other_light' : ['sum'],
                'Count_Other_moderate' : ['sum'],
                'Count_Other_severe' : ['sum'],
                'Count_Flood' : ['sum'],
                'Count_Storm' : ['sum'],
                'Count_Wind' : ['sum'],
                'Count_Fire' : ['sum'],
               }

    # Compute data
    data = {}
    for range_ in ['1st', '2nd', '3rd', '4th'] :
        # Loop over fips_codes
        for fips in tqdm(df['fips'].unique(), position=0, leave=True):
            # Early stopping
            if fips not in dict_fips :
                continue
            # Get neighbours fips
            neigh_fips = dict_fips[fips][f'neighbours_{range_}']
            # Find neigh_fips actually present in df
            neigh_fips = list(fips_set.intersection(neigh_fips))
            # If no neighbours is in df, we can pass to next fips
            if len(neigh_fips) == 0 :
                continue
            # Select rows corresponding to the neighbours and group them by date
            df_tmp = df.loc[neigh_fips].groupby(cols_date).agg(dict_agg).reset_index()
            # Flatten column names
            df_tmp.columns = cols_date + ['_'.join(i).rstrip('_') for i in df_tmp.columns[len(cols_date):]]
            # Complete df
            for col in df_tmp.columns[len(cols_date):] :
                df.loc[fips, f"{col}_among_{range_}_neighbours"] = df_tmp[col].values

    # --------------------------------------------------------------------------------------
    # Group by ALL neighbours (+ main fips)
    
    # SUM
    for k in ['outages_sum',
              'outages_outbreak_sum',
              'Severe_events_count_sum',
              'Population_sum',
              # WORDS
              'MAGNITUDE_sum',
              'WORDS_wind_and_storm_sum',
              'WORDS_downed_trees_sum',
              'WORDS_heavy_damage_sum',
              'WORDS_wires_sum',
              'WORDS_heat_and_fire_sum',
              # Count
              'Count_Other_light_sum',
              'Count_Other_moderate_sum',
              'Count_Other_severe_sum',
              'Count_Flood_sum',
              'Count_Storm_sum',
              'Count_Wind_sum',
              'Count_Fire_sum',
             ] :
        cols = [k.replace('_sum', ''),
                f'{k}_among_1st_neighbours',
                f'{k}_among_2nd_neighbours',
                f'{k}_among_3rd_neighbours',
                f'{k}_among_4th_neighbours',]
        df[f'{k}_among_ALL_neighbours'] = df[cols].sum(axis=1)
      
    # Return
    return df
    
# -----------------------------------------------------------------------------------------------------


def get_word_counter(df, col, delete_stopwords=True) :
    """
    Generates a word frequency counter from a specified column in the DataFrame.

    Parameters:
        - df (pd.DataFrame): Input DataFrame containing the text data.
        - col (str): Name of the column containing the text data to be processed.
        - delete_stopwords (bool, optional, default=True): If True, removes common English stopwords from the word list.

    Returns:
        Counter: A Counter object containing the frequency of each word in the specified column.
    """

    # Flatten all words into one list
    word_list = ' '.join(df[col].dropna().values).split()
    
    # Delete stopwords
    if delete_stopwords :
        from nltk.corpus import stopwords
        english_stopwords = set(stopwords.words('english'))
        filtered_words = [word for word in word_list if word not in english_stopwords]
    
    # Return
    return Counter(filtered_words)

# -----------------------------------------------------------------------------------------------------

def plot_most_frequent_words(counter, n_words=50, figsize=(10, 10)) :
    """
    Plots the most frequent words from a Counter object as a bar chart.

    Parameters:
        - counter (Counter): A Counter object containing word frequencies.
        - n_words (int, optional, default=50): Number of top words to display.
        - figsize (tuple, optional, default=(10, 10)): Size of the figure (width, height) in inches.

    Returns:
        None: Displays the bar plot of the most frequent words.
    """

    # Select most frequent words
    top_words = counter.most_common(n_words)
    words, freqs = zip(*top_words)
    
    # Barplot
    plt.figure(figsize=figsize)
    sns.barplot(x=list(freqs), y=list(words), palette="magma")
    plt.title(f"Top {n_words} Most Frequent Words")
    plt.xlabel("Frequency")
    plt.show()
   
# -----------------------------------------------------------------------------------------------------

def plot_words_by_category(counter, categories, figsize=(10, 17)):
    """
    Plots the frequency of words categorized by predefined categories as a horizontal bar chart.

    Parameters:
        - counter (Counter): A Counter object containing word frequencies.
        - categories (dict): A dictionary mapping category names to lists of words belonging to each category.
        - figsize (tuple, optional, default=(10, 17)): Size of the figure (width, height) in inches.

    Returns:
        None: Displays the horizontal bar plot of word frequencies by category.
    """

    # Color 
    category_colors = {
                        'WORDS_wind_and_storm': 'black',
                        'WORDS_downed_trees': 'green',        
                        'WORDS_heavy_damage': 'red',      
                        'WORDS_wires': 'gray',   
                        'WORDS_water_and_flood': 'blue',
                        'WORDS_heat_and_fire': 'orange',            
                        'WORDS_snow': 'lightblue',
                       }
    
    # Collect word frequencies by category
    word_freqs = []
    for category, words in categories.items():
        for word in words:
            freq = counter[word]
            if freq > 0:
                word_freqs.append((word, freq, category))
    
    # Sort by frequency
    word_freqs = sorted(word_freqs, key=lambda x: x[1], reverse=True)
    
    # Plot
    plt.figure(figsize=(8, 16))
    for word, freq, category in word_freqs:
        plt.barh(word, freq, color=category_colors[category])
        
    plt.gca().invert_yaxis()
    plt.xlabel("Frequency")
    plt.title("Keyword Frequencies by Category")
    plt.legend(handles=[plt.Rectangle((0,0),1,1,color=category_colors[c], label=c) for c in categories], loc='center right')
    plt.tight_layout()
    plt.show()
    
# -----------------------------------------------------------------------------------------------------

def get_days_since_last_value_greater_than_thr(df, col, col_to_create, thr):
    """
    Calculates the number of days since the last value was greater than (or equal) to a given threshold.

    Parameters:
        - df (pd.DataFrame): Input DataFrame containing the data.
        - col (str): Name of the column to evaluate against the threshold.
        - col_to_create (str): Name of the column to create with the calculated days since the last value greater than the threshold.
        - thr (float): Threshold value to compare against the values in the specified column.

    Returns:
        pd.DataFrame: DataFrame with an additional column indicating the number of days since the last value greater than or equal to the threshold.
    """
    # Find index of positive values
    positive_mask = (df[col] >= thr)

    # Initialize column
    df[col_to_create] = (df[col] < thr).astype(int)

    # Create cumulative count
    cumcount = df.groupby('fips')[col_to_create].cumcount()

    # Store values (Substract values where there were positive values)
    df[col_to_create] = cumcount - cumcount.where(positive_mask).ffill().fillna(0).astype(int)

    # Negative values are caused by ffill on first value of a group, when they are below thr
    df[col_to_create] = df[col_to_create].clip(-1, None)

    # Return
    return df

def get_days_since_last_positive_value(df, col, col_to_create):
    """
    Calculates the number of days since the last positive value in a specified column.

    Parameters:
        - df (pd.DataFrame): Input DataFrame containing the data.
        - col (str): Name of the column to evaluate for positive values.
        - col_to_create (str): Name of the column to create with the calculated days since the last positive value.

    Returns:
        pd.DataFrame: DataFrame with an additional column indicating the number of days since the last positive value.
    """

    return get_days_since_last_value_greater_than_thr(df, col, col_to_create, thr=1e-5)

def get_days_before_next_value_greater_than_thr(df, col, col_to_create, thr):
    """
    Calculates the number of days before the next value in a specified column is greater than or equal to a given threshold.

    Parameters:
        - df (pd.DataFrame): Input DataFrame containing the data.
        - col (str): Name of the column to evaluate against the threshold.
        - col_to_create (str): Name of the column to create with the calculated days before the next value greater than the threshold.
        - thr (float): Threshold value to compare against the values in the specified column.

    Returns:
        pd.DataFrame: DataFrame with an additional column indicating the number of days before the next value greater than or equal to the threshold.
    """


    # Find index of positive values
    positive_mask = (df[col] >= thr)

    # Initialize column
    df[col_to_create] = (df[col] < thr).astype(int)

    # Create cumulative count
    cumcount = df.groupby('fips')[col_to_create].cumcount()

    # Store values (Substract values where there were positive values)
    df[col_to_create] = cumcount.where(positive_mask).bfill().fillna(0).astype(int) - cumcount

    # Negative values are caused by bfill on lst values of a group, when they are below thr
    df[col_to_create] = df[col_to_create].clip(-1, None)

    # Return
    return df

def get_days_before_next_positive_value(df, col, col_to_create):
    """
    Calculates the number of days before the next positive value in a specified column.

    Parameters:
        - df (pd.DataFrame): Input DataFrame containing the data.
        - col (str): Name of the column to evaluate for positive values.
        - col_to_create (str): Name of the column to create with the calculated days before the next positive value.

    Returns:
        pd.DataFrame: DataFrame with an additional column indicating the number of days before the next positive value.
    """

    return get_days_before_next_value_greater_than_thr(df, col, col_to_create, thr=1e-5)
    
# -----------------------------------------------------------------------------------------------------

def plot_timelapse(df, col, start_date, end_date, plot_only_fips_in_df=True,
                   range_color=None,
                   gpd_url=gpd_url):

    """
    Generates an animated choropleth map to visualize the timelapse of a specified column's values over a date range.

    Parameters:
        - df (pd.DataFrame): Input DataFrame containing the data.
        - col (str): Name of the column to visualize.
        - start_date (str or datetime-like): Start date for the timelapse.
        - end_date (str or datetime-like): End date for the timelapse.
        - plot_only_fips_in_df (bool, optional, default=True): If True, plots only the FIPS codes present in the DataFrame.
        - range_color (tuple, optional): Tuple specifying the range of colors for the choropleth map. If None, the range is set to the minimum and maximum values of the specified column.
        - gpd_url (str, optional): URL or path to the GeoPandas shapefile.

    Returns:
        None: Displays the animated choropleth map using Plotly.
    """


    # Load shapefile
    gdf = gpd.read_file(gpd_url)
    
    # Filter continental US
    gdf = gdf[~gdf['STATE_NAME'].isin(['Puerto Rico', 'Alaska', 'Hawaii'])]
    gdf['GEOID'] = gdf['GEOID'].astype(int)
    
    # Simplify geometry for performance
    gdf = gdf[gdf['GEOID'].isin(df['fips'].unique())].reset_index(drop=True)
    gdf = gdf[['GEOID', 'geometry']].copy()
    gdf = gdf.set_index('GEOID')
    gdf_json = json.loads(gdf.to_json())
    
    # Prepare data
    df_plot = df[(df['date'].between(start_date, end_date)) & (~df[col].isna())].reset_index(drop=True)
    if 'hour' in df_plot :
        df_plot['date'] = pd.to_datetime(df_plot['date']) + pd.to_timedelta(df_plot['hour'], unit='h')

    # Range color
    if range_color is None :
        range_color = (df_plot[col].min(), df_plot[col].max())
        
    # Build Plotly figure
    fig = px.choropleth(df_plot,
                        geojson=gdf_json,
                        locations='fips',
                        color=col,
                        animation_frame='date',
                        color_continuous_scale=['#e5eef8', '#A7C7E7', '#E85E4C'],
                        range_color=range_color,
                        scope='usa',
                        labels={'preds': 'Prediction',
                                'outages':'Outages',
                                'outages_outbreak':'Outages Outbreak'},
                        )
    
    # Layout tuning
    fig.update_layout(
        title='Predictions by Hour',
        geo=dict(showlakes=False, lakecolor='LightBlue'),
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    
    fig.show()

# -----------------------------------------------------------------------------------------------------

from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
def get_ellipse_caracteristics_around_points(group_coords, weights, exclude_outliers_from_npoints=float('inf')):
    """
    Computes the characteristics of an ellipse that best fits a set of weighted points, optionally excluding outliers.

    Parameters:
        - group_coords (np.ndarray): Array of coordinates (latitude, longitude) of the points.
        - weights (np.ndarray): Array of weights corresponding to each point.
        - exclude_outliers_from_npoints (float, optional, default=inf): Threshold number of points to trigger outlier exclusion. If the number of points is greater than or equal to this threshold, outliers are excluded based on Mahalanobis distance or Euclidean distance.

    Returns:
        tuple: A tuple containing the following ellipse characteristics:
            - lat (float): Latitude of the ellipse centroid.
            - lon (float): Longitude of the ellipse centroid.
            - width (float): Width of the ellipse (major axis length).
            - height (float): Height of the ellipse (minor axis length).
            - angle (float): Orientation angle of the ellipse in degrees.
    """


    # If the cluster has a lot of points, remove outliers
    if len(group_coords) >= exclude_outliers_from_npoints:
        try: # try to use mahalanobis distance (better than euclidian)
            mean = np.average(group_coords, axis=0, weights=weights)
            cov = np.cov(group_coords.T, aweights=weights)
            inv_cov = inv(cov)
            dists = np.array([mahalanobis(pt, mean, inv_cov) for pt in group_coords])
        except: # covariance not invertible -> use euclidian
            lat_c = np.average(group_coords[:, 0], weights=weights)
            lon_c = np.average(group_coords[:, 1], weights=weights)
            dists = np.sqrt((group_coords[:, 0] - lat_c)**2 + (group_coords[:, 1] - lon_c)**2)
        # Filter outliers
        keep_mask = dists <= np.quantile(dists, 0.9)
        group_coords = group_coords[keep_mask]
        weights = weights[keep_mask]
    
    # --- 1. Cluster centroid (ponderated latitude and longitude) ---
    lat = np.average(group_coords[:, 0], weights=weights)
    lon = np.average(group_coords[:, 1], weights=weights)
    
    # --- 2. Weighted covariance matrix (to extract orientation, height and width) ---
    centered = group_coords - ((lat, lon))
    cov = np.cov(centered.T, aweights=weights)
    
    # --- 3. Eigenvalues/eigenvectors: ellipse axes & orientation ---
    eigvals, eigvecs = np.linalg.eigh(cov)  # sorted, small to large
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    
    # --- 4. Axis lengths & Orientation ---
    scale = 2  # for 2 standard deviations (≈95% of poitns are inside the ellipse)
    width, height = 2 * scale * np.sqrt(eigvals)  # major & minor axis lengths
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))  # major axis

    # Return
    return lat, lon, width, height, angle
    
# -----------------------------------------------------------------------------------------------------

from sklearn.cluster import DBSCAN
def compute_centroids(df, col='outages', threshold=300, dbscan_eps=0.5, dbscan_min_samples=3):
    """
    Computes the centroids of clusters formed by points using DBSCAN clustering.

    Parameters:
        - df (pd.DataFrame): Input DataFrame containing the data.
        - col (str, optional, default='outages'): Name of the column to evaluate for thresholding.
        - threshold (float, optional, default=300): Threshold value to filter rows with high values in the specified column.
        - dbscan_eps (float, optional, default=0.5): The maximum distance between two samples for one to be considered as in the neighborhood of the other in DBSCAN.
        - dbscan_min_samples (int, optional, default=3): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point in DBSCAN.

    Returns:
        dict: A dictionary where keys are timestamps and values are lists of centroids. Each centroid is a dictionary containing:
            - lat (float): Latitude of the centroid.
            - lon (float): Longitude of the centroid.
            - width (float): Width of the ellipse around the cluster.
            - height (float): Height of the ellipse around the cluster.
            - angle (float): Orientation angle of the ellipse in degrees.
            - weights (float): Mean weight of the points in the cluster.
            - cluster_size (int): Number of points in the cluster.
    """

    # Sort df
    df = df.sort_values(by = ['fips', 'date', 'hour']).reset_index(drop=True)
    
    # Put fips as index to select row faster
    df.index = df['fips'].values
    
    # Get all timestamps
    df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    n_hours_in_df_per_fips = df['timestamp'].nunique() # 24 * df['date'].nunique()
    timestamps = list(map(str, sorted(df['timestamp'].unique())))
    
    # Initiate result
    result = {}

    # Iterate over datetime
    for i, ts in enumerate(tqdm(timestamps, total=n_hours_in_df_per_fips)):
    
        # Select a row per fips (for the same date+hour)
        df_hour = df.iloc[i::n_hours_in_df_per_fips]
    
        # Filter rows with high number of columns
        df_hour = df_hour[df_hour[col] >= threshold]
        if df_hour.empty :
            result[str(ts)] = None
            continue
    
        # Get coordinates
        coords  = df_hour[['latitude', 'longitude']].values
        numbers = df_hour[col].values

        # Clip to avoid one fips having too much weight
        #numbers = np.clip(numbers, 0, 15000)
        
        # Run DBSCAN (eps = 0.1 → ~11 km radius)
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords)
        labels = db.labels_
    
        # Compute centroïds
        centroids = []
        for label in set(labels):
    
            # Skip noise
            if label == -1: continue
        
            # Index of current cluster
            idx = (labels == label)
            group_coords = coords[idx]
            weights = numbers[idx]

            # Extract informations about the ellipse around the cluster
            lat, lon, width, height, angle = get_ellipse_caracteristics_around_points(group_coords,
                                                                                      weights,
                                                                                      exclude_outliers_from_npoints=10)
        
            #Create a dictionnary with all information
            centroid = {
                        "lat" : round(lat, 6),
                        "lon" : round(lon, 6),
                        "width"  : round(width, 6),
                        "height" : round(height, 6),
                        "angle"  : round(angle, 6),
                        "weights" : np.mean(weights),
                        "cluster_size" : len(weights),
                       }
            centroids.append(centroid)
    
        # Store result
        result[ts] = centroids

    # Return
    return result

# -----------------------------------------------------------------------------------------------------

def get_centroids_evolution(centroids, n_hours_ago=4, max_km=100):
    """
    Tracks the evolution of centroids over time, identifying movement vectors between consecutive timestamps.

    Parameters:
        - centroids (dict): A dictionary where keys are timestamps and values are lists of centroids. Each centroid is a dictionary containing geographical and cluster information.
        - n_hours_ago (int, optional, default=4): The number of previous hours to consider for tracking the evolution of centroids.
        - max_km (float, optional, default=100): The maximum distance in kilometers to consider a centroid as moving from one timestamp to another.

    Returns:
        dict: A dictionary where keys are timestamps and values are lists of movement vectors. Each vector is a dictionary containing:
            - lat (float): Latitude of the current centroid.
            - lon (float): Longitude of the current centroid.
            - lat_diff (float): Difference in latitude per hour.
            - lon_diff (float): Difference in longitude per hour.
            - width (float): Width of the ellipse around the current cluster.
            - height (float): Height of the ellipse around the current cluster.
            - angle (float): Orientation angle of the ellipse in degrees.
            - weights_diff (float): Difference in weights per hour.
            - weights (float): Mean weight of the points in the current cluster.
            - cluster_size (int): Number of points in the current cluster.
            - cluster_size_diff (int): Difference in cluster size from the previous timestamp.
    """

    # Initialize
    vectors = {}
    
    # Loop over pairs of consecutive centroids (separated by n_hours_diff)
    keys = list(centroids.keys())
    for i in range(len(keys)):

        # Extract timestamps
        ts2 = keys[i]
        if centroids[ts2] is None :
            vectors[ts2] = None
            continue

        # Initialize found vectors for t2
        found_vectors = []

        # For all centroïds of ts1
        for c2 in centroids[ts2] :

            # Initialize closest centroïds from previous hours
            closest, min_dist = None, float('inf')
            vector = None

            # Look at all previous centroïds to find the closest to c2
            for n_hours_diff in range(1, n_hours_ago+1):

                # Break
                if (i-n_hours_diff < 0) :
                    break
    
                # Extract timestamps
                ts1 = keys[i-n_hours_diff]
        
                # Continue
                if centroids[ts1] is None :
                    continue
            
                # Find closest centroids of centroid(ts2)
                for c1 in centroids[ts1] :
                    dist = geodesic((c1['lat'], c1['lon']), (c2['lat'], c2['lon'])).km
                    if dist < min_dist and dist <= max_km*n_hours_diff:
                        min_dist = dist
                        closest = c1

                # If a close centroïd has been found
                if closest :
                    lat_diff = round(c2['lat'] - closest['lat'], 6)
                    lon_diff = round(c2['lon'] - closest['lon'], 6)
                    weights_diff = int(c2['weights'] - c1['weights'])
                    ponderated_angle = (4*c2['angle'] + (n_hours_ago-n_hours_diff)*closest['angle'])/(4+n_hours_ago-n_hours_diff)
                    # Store vector if it's moving
                    if lat_diff!=0 or lon_diff!=0:
                        vector = {
                                    "lat" : c2['lat'],
                                    "lon" : c2['lon'],
                                    "lat_diff" : lat_diff / n_hours_diff,
                                    "lon_diff" : lon_diff / n_hours_diff,
                                    "width"  : c2['width'],
                                    "height" : c2['height'],
                                    "angle" : c2['angle'],
                                    "weights_diff" : weights_diff / n_hours_diff,
                                    "weights" : c2['weights'],
                                    "cluster_size" : c2['cluster_size'],
                                    "cluster_size_diff" : c2['cluster_size'] - c1['cluster_size'],
                                 }
                        break

            # Save vector
            if vector is not None :
                found_vectors.append(vector)
                    
        # Store information
        vectors[ts2] = found_vectors

    # Return
    return vectors
    
# -----------------------------------------------------------------------------------------------------

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrow
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse

def plot_evolution(df,
                   date,
                   hours_list,
                   vectors = [],
                   color_by="outages",
                   fips=[],
                   cmap=custom_cmap,
                   figsize = (10, 6),
                   arrow_size = 1,
                   use_quantile_as_max_for_color_normalization=0.99,
                   reverse_colormap=False,
                   fillna_value = float('inf'),
                   plot_ellipses=False,
                   min_weight_to_plot_arrow=-float('inf'),
                   min_cluster_size_to_plot_ellipses=-float('inf'),
                   gpd_url=gpd_url,
                  ):

    """
    Plots the evolution of a geospatial variable over multiple hours using a US counties map.

    Parameters
    ----------
    df : pd.DataFrame
        Main dataframe containing at least ['fips', 'date', 'hour', color_by].
    date : str
        Date to filter the data (format: 'YYYY-MM-DD').
    hours_list : list of int
        List of hours (0–23) to visualize.
    vectors : dict, optional
        Dictionary mapping timestamps to flow vectors. Each vector must include
        'lat', 'lon', 'lat_diff', 'lon_diff', 'width', 'height', 'angle', and 'numbers'.
    color_by : str, default "outages"
        Column in df to use for coloring the counties.
    fips : list of int, optional
        List of FIPS codes to restrict plotting to specific counties. If empty, all available are used.
    cmap : matplotlib colormap, default custom_cmap
        Colormap used to color counties based on the `color_by` variable.
    figsize : tuple, default (10, 6)
        Size of the matplotlib figure.
    arrow_size : float, default 1
        Scaling factor for arrow lengths representing vector magnitude.
    use_quantile_as_max_for_color_normalization : float, default 0.99
        If > 0, uses this quantile to set the upper bound of the colormap scale for robustness to outliers.
    reverse_colormap : bool, default False
        Whether to reverse the colormap.
    fillna_value : float, default float('inf')
        Value to use to fill missing values in `color_by`. If set to np.inf, no filling is done.
    plot_ellipses : bool, default False
        If True, overlays ellipses defined in `vectors` data to visualize shape and orientation.
    min_weight_to_plot_arrow : float, optional
        Minimum 'weight' value for a vector to be plotted (used to filter small/local arrows).
    min_cluster_size_to_plot_ellipses : float, optional
        Minimum 'cluster size' value for a vector ellipse to be plotted (used to filter small/local ellipses).
    gpd_url : str
        URL to shapefile for USA map plotting.

    """

    # If a list of fips has been provided
    if len(fips)>0:
        fips_to_plot = fips
    else :
        fips_to_plot = sorted(df['fips'].unique())

    # Load US counties shapefile
    counties = gpd.read_file(gpd_url)
    counties = counties[~counties['STATE_NAME'].isin(['Puerto Rico', 'Alaska', 'Hawaii'])]
    counties['GEOID'] = counties['GEOID'].astype(int)
    counties = counties[counties['GEOID'].isin(fips_to_plot)].reset_index(drop=True)

    # Create a temporary df
    mask = (df['date']==date) & (df['hour'].isin(hours_list)) & (df['fips'].isin(fips_to_plot))
    df_tmp = df.loc[mask, ['fips', 'date', 'hour', color_by]].reset_index(drop=True)

    # Pass
    if df_tmp.empty:
        print("Nothing to plot !")
        return
    
    # Fill nan
    if np.isfinite(fillna_value) :
        df_tmp[color_by].fillna(fillna_value, inplace=True)

    # Set up color normalization
    vmin, vmax = df_tmp[color_by].min(), df_tmp[color_by].max()
    if use_quantile_as_max_for_color_normalization>0:
        vmax = np.quantile(df_tmp[color_by].dropna(), q=use_quantile_as_max_for_color_normalization)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Reverse colormap
    if reverse_colormap:
        cmap = cmap.reversed()

    # Subplot
    fig, axes = plt.subplots(1, len(hours_list), figsize=figsize)

    # To avoid problems if only 1 hour is provided
    if len(hours_list)==1:
        axes = [axes, None]

    # Global title
    
    fig.suptitle(color_by + f': {date}', fontsize=14, y=0.9 + 0.06*(len(hours_list)<=3))

    for i, hour in enumerate(hours_list) :
        
        # Filter date
        df_plot = df_tmp[(df_tmp['hour']==hour)].reset_index(drop=True)
        df_plot['fips'] = df_plot['fips'].astype(int)
        
        # Merge with preds
        counties_merged = pd.merge(counties,
                                   df_plot[['fips', color_by]].rename(columns={'fips': 'GEOID'}),
                                   on='GEOID',
                                   how='left')
    
        # Plot
        counties_merged.plot(column=color_by, cmap=cmap, linewidth=0.5, ax=axes[i], edgecolor='gray',
                             norm=norm)
    
        # Set title
        axes[i].set_title(f"Hour {hour}", fontsize=12)
        axes[i].axis('off')

        # ---------------------------------------

        # Plot arrows
        if vectors :

            # Get timestamp
            timestamp = str(pd.to_datetime(date) + pd.to_timedelta(hour, unit='h'))
            if timestamp not in vectors :
                continue
                
            # Prepare arrow data
            arrows = []
            for v in vectors[timestamp]:
                # Continue if the vector is a little local one
                if v['weights'] < min_weight_to_plot_arrow: continue
                # Plot arrow
                dx = v["lon_diff"] * arrow_size
                dy = v["lat_diff"] * arrow_size
                area  = np.clip((v["width"] * v["height"]) / 40, 0.08, 0.3)
                color = plt.cm.Greys(np.clip(v['weights']/5000, 0.15, 1))  # "black"
                arrow = FancyArrow(v["lon"], v["lat"], dx, dy, width=area, length_includes_head=False, color=color, edgecolor="black")
                arrows.append(arrow)

                # Plot ellipses
                if plot_ellipses and v['cluster_size'] >= min_cluster_size_to_plot_ellipses:
                    linewidth = np.clip(v['weights']/1000, 0.8, 2)
                    ellipse = Ellipse(xy=np.array((v["lon"], v["lat"])), width=v["width"], height=v["height"],
                                      angle=v["angle"], edgecolor="black", fc='none', lw=linewidth, label='Reconstructed ellipse',
                                      linestyle='--',
                                     )
                    axes[i].add_patch(ellipse)

            # Add arrows
            if arrows :
                arrow_collection = PatchCollection(arrows,
                                                   facecolor=[a.get_facecolor() for a in arrows],
                                                   edgecolor="black")
                axes[i].add_collection(arrow_collection)

    # ---------------------------------------
    # Shared colorbar on bottom
    cbar_ax = fig.add_axes([0.25, 0.02+0.07*(len(hours_list)>3), 0.5, 0.02])  # [left, bottom, width, height]
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(color_by, fontsize=12)

    # Show
    plt.tight_layout()
    plt.show()
    
# -----------------------------------------------------------------------------------------------------

def merge_vectors(vectors_1, vectors_2):
    """
    Merges two sets of movement vectors, combining similar vectors based on predefined thresholds for each attribute.

    Parameters:
        - vectors_1 (dict): A dictionary where keys are timestamps and values are lists of movement vectors.
        - vectors_2 (dict): A dictionary where keys are timestamps and values are lists of movement vectors.

    Returns:
        dict: A dictionary where keys are timestamps and values are lists of merged movement vectors. Each vector is a dictionary containing:
            - lat (float): Latitude of the current centroid.
            - lon (float): Longitude of the current centroid.
            - lat_diff (float): Difference in latitude per hour.
            - lon_diff (float): Difference in longitude per hour.
            - width (float): Width of the ellipse around the current cluster.
            - height (float): Height of the ellipse around the current cluster.
            - angle (float): Orientation angle of the ellipse in degrees.
            - weights_diff (float): Difference in weights per hour.
            - weights (float): Mean weight of the points in the current cluster.
            - cluster_size (int): Number of points in the current cluster.
            - cluster_size_diff (int): Difference in cluster size from the previous timestamp.
    """

    # Vectors Keys (timestamps)
    keys = set(list(vectors_1.keys()) + list(vectors_2.keys()))

    # Combine vectors
    merged_vectors = {}
    for key in keys :
        L1, L2 = [], []
        if key in vectors_1 and vectors_1[key] is not None:
            L1 = vectors_1[key]
        if key in vectors_2 and vectors_2[key] is not None:
            L2 = vectors_2[key]
        # Add vectores
        merged_vectors[key] = L1 + L2

    # Thr for each column
    thr_dict = {'lat': 0.25,
                'lon': 0.25,
                'lat_diff': 0.5,
                'lon_diff': 0.5,
                'width': 0.8,
                'height': 0.8,
                'angle': 5, # in degrees
                'cluster_size': 30}

    # Initialize result
    result = {}
    
    # Fusion of similar vectors
    for key, vectors_list in merged_vectors.items():

        # Initialize list of fusionized vectors
        fusioned = []

        # Index of vectors in vectors_list that have not been fusioned yet
        remaining_index = set(range(len(vectors_list)))

        # For all vectors, find similar ones
        while len(remaining_index) > 0 :
            i1 = remaining_index.pop() # Delete one element
            v1 = vectors_list[i1]
            similar_vect_idx = [i1]

            # Find similar vectors to v1
            for i2 in remaining_index :
                v2 = vectors_list[i2]
                #if i1==0 and i2==2:
                #    print([(k, abs(v1[k] - v2[k])) for k in thr_dict.keys()])
                if np.all([(abs(v1[k] - v2[k]) <= thr_dict[k]) for k in thr_dict.keys()]):
                    similar_vect_idx.append(i2) # Found similar vector !

            # Delete similar vectors from remaining_index
            remaining_index = remaining_index.difference(similar_vect_idx)

            # Fusion similar vector (keep the biggest)
            if len(similar_vect_idx) >= 2:
                vectors = [vectors_list[i] for i in similar_vect_idx]
                vectors = sorted(vectors, key = lambda x : (x['cluster_size'], x['width']*x['height']), reverse=True)
                fusioned.append(vectors[0])
            else :
                fusioned.append(v1)

        # Update result
        result[key] = fusioned
  
    # Return
    return result

# -----------------------------------------------------------------------------------------------------

def compute_dist_points_to_ellipse(lat, lon, lat_c, lon_c, width, height, angle_deg):
    """
    Computes the distances of points to an ellipse and determines whether the points are inside the ellipse.

    Parameters:
        - lat (float or array-like): Latitude(s) of the point(s).
        - lon (float or array-like): Longitude(s) of the point(s).
        - lat_c (float): Latitude of the ellipse center.
        - lon_c (float): Longitude of the ellipse center.
        - width (float): Width of the ellipse (major axis length).
        - height (float): Height of the ellipse (minor axis length).
        - angle_deg (float): Orientation angle of the ellipse in degrees.

    Returns:
        tuple: A tuple containing:
            - inside (np.ndarray): Boolean array indicating whether each point is inside the ellipse.
            - dist_to_boundary (np.ndarray): Distance of each point to the ellipse boundary in kilometers.
            - dist_to_center (np.ndarray): Distance of each point to the ellipse center in kilometers.
    """

    # Convert inputs to numpy arrays
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    # Convert angle to radians
    theta = np.radians(angle_deg)

    # Translate points to center
    dx = lon - lon_c
    dy = lat - lat_c

    # Distance to center (straight lines)
    dist_to_center = np.sqrt(dx**2 + dy**2)

    # Rotate points to align with ellipse axes
    x_rot = dx * np.cos(theta) + dy * np.sin(theta)
    y_rot = -dx * np.sin(theta) + dy * np.cos(theta)

    # Normalize to ellipse axes
    nx = x_rot / (width / 2)
    ny = y_rot / (height / 2)

    # Ellipse value
    ellipse_val = nx**2 + ny**2
    inside = ellipse_val <= 1

    # Distance to boundary (in degree)
    a = width / 2
    b = height / 2
    dist_to_boundary = np.abs((np.sqrt(ellipse_val) - 1) * np.sqrt(a**2 + b**2)) # delete abs() to get pos/neg depending on inside/outside ellipse

    # Convert distance in degree to km (1 degree ≈ 111 km)
    dist_to_center *= 111
    dist_to_boundary *= 111

    # Return
    return inside, dist_to_boundary, dist_to_center

# -----------------------------------------------------------------------------------------------------

def forecast_hit_zone_of_vectors(df, df_latlong, vectors_outages, dict_fips):
    """
    Forecasts the impact zones of vectors by projecting their future characteristics and calculating the impact on nearby regions.

    Parameters:
        - df (pd.DataFrame): Input DataFrame containing the data.
        - df_latlong (pd.DataFrame): DataFrame containing latitude and longitude information for each FIPS code.
        - vectors_outages (dict): A dictionary where keys are timestamps and values are lists of movement vectors.
        - dict_fips (dict): A dictionary containing FIPS codes and their neighboring FIPS codes.

    Returns:
        pd.DataFrame: DataFrame with additional columns indicating the projected impact of vectors on nearby regions.
    """

    # Sort df
    df = df.sort_values(by = ['fips', 'date', 'hour']).reset_index(drop=True)
    
    # Get all timestamps
    df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    n_hours_in_df_per_fips = df['timestamp'].nunique() # 24 * df['date'].nunique()
    timestamps = sorted(map(str, df['timestamp'].unique()))
    
    # Set fips as index to accelerate search
    df.index = df['fips'].astype(str) + df['timestamp'].astype(str)
    
    # Loop over all timestamps
    for i, ts in enumerate(tqdm(timestamps)):
    
        # Continue
        if ts not in vectors_outages or vectors_outages[ts] is None:
            continue
    
        # Loop over some hours we want to make predictions at
        for n_hours in [3, 6]:
    
            # Initialize a list of df for the current timestamp
            df_ts = []
        
            # Loop over all vectors of the current timestamp
            for v in vectors_outages[ts]:
            
                # Projection of future caracteristics
                projected_lat = v['lat'] + n_hours*v['lat_diff']
                projected_lon = v['lon'] + n_hours*v['lon_diff']
    
                # Create an unit-less mesure to caracterize the impact (depend on the mean weight and the cluster size)
                weight      = np.clip(v['weights'] / 600, 0, 10)
                weight_diff = np.clip(n_hours*v['weights_diff']/250, -3, 3)
                cluster_size_diff = np.clip(v['cluster_size_diff']/2, -1, 10) # Big increase is meaningful, but decrease is not really
                impact = round(weight + weight_diff + cluster_size_diff, 1)
    
                # If the impact increases, we increase the size of the projected ellipse
                projected_width  = v['width']
                projected_height = v['height']
                if impact > 0:
                    projected_width += n_hours*0.15
                    projected_height += n_hours*0.15
                
                # Compute distances to the center and the boundary of the projected cluster (ellipse)
                is_inside, dist_to_boundary, dist_to_center = compute_dist_points_to_ellipse(df_latlong['lat'].values,
                                                                                             df_latlong['lon'].values,
                                                                                             projected_lat, # projected latitude of ellipse center
                                                                                             projected_lon, # projected longitude of ellipse center
                                                                                             projected_width,
                                                                                             projected_height,
                                                                                             v['angle'])
                df_latlong[f'projected_distance_to_centroid_center_in_{n_hours}hours'] = dist_to_center
    
                # Add close neighbours of the ellipse boundaries
                #is_inside = is_inside | (dist_to_boundary <= 20) # distance in km
    
                # Get fips inside the ellipse and add their 1st neighbours
                impacted_fips = list(df_latlong.loc[is_inside, "fips"].unique())
                adjacent_fips = [dict_fips[fips]['neighbours_1st'] for fips in impacted_fips]
                impacted_fips += list(set([f for ff in adjacent_fips for f in ff]))
    
                # Find area touched by the projection, which are all positions (x, y) such that : (abs(x-lat) < lat_width) and (abs(y-lon) < lon_width)
                df_impacted_fips = df_latlong[df_latlong['fips'].isin(impacted_fips)].reset_index(drop=True)
    
                # Select columns
                df_impacted_fips = df_impacted_fips[['fips',
                                                     f'projected_distance_to_centroid_center_in_{n_hours}hours',
                                                   ]]
    
                # Add columns
                for k in ['weights', 'weights_diff', 'cluster_size', 'cluster_size_diff']:
                    df_impacted_fips[f'vector_{k}'] = v[k]
                df_impacted_fips[f'projected_fips_impacted_in_{n_hours}hours'] = len(impacted_fips)
    
                # Impact : the closer a point is to the cluster center, the bigger the impact
                coef_dists = 1 + (df_impacted_fips[f'projected_distance_to_centroid_center_in_{n_hours}hours'] / 150)
                coef_dists = np.clip(coef_dists, 1, 3)
                df_impacted_fips[f'projected_impact_in_{n_hours}hours'] = np.clip(impact, 0, None) / np.clip(coef_dists**2, 1, 4)
    
                # Add information to global df (for the current timestamp)
                df_ts.append(df_impacted_fips.copy())
    
            # Concat all df_ts
            if df_ts:
    
                # Concat all df_ts
                df_ts = pd.concat(df_ts).reset_index(drop=True)
    
                # Keep only 1 row by fips (corresponding to the strongest storm)
                idx = df_ts.groupby('fips')[f'projected_impact_in_{n_hours}hours'].idxmax()
                df_ts = df_ts.loc[idx].reset_index(drop=True)
    
                # Fill up df column
                index = df_ts['fips'].astype(str) + ts
                for col in df_ts :
                    if col not in ['fips', 'date', 'hour']:
                        df.loc[index, col] = df_ts[col].values

    # Manually create columns if nothing was done in the loops (for instance if no vectors has been found)
    cols = ['vector_weights', 'vector_weights_diff', 'vector_cluster_size', 'vector_cluster_size_diff']
    for n_hours in [3, 6]:
        cols += [f'projected_distance_to_centroid_center_in_{n_hours}hours',
                 f'projected_fips_impacted_in_{n_hours}hours',
                 f'projected_impact_in_{n_hours}hours'
                ]
    for k in cols :
        if k not in df :
            df[k] = None
    
    # Delete timestamp
    df = df.drop(columns = ['timestamp']).reset_index(drop=True)

    # Return
    return df

# -----------------------------------------------------------------------------------------------------

def plot_preds_on_map(df,
                      col,
                      figsize=(10, 8),
                      cmap=custom_cmap,
                      gpd_url=gpd_url,
                     ):
    """
    Plots a FIPS-based prediction map of continental US counties with a custom color map and legend.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'fips' (int or str) and 'preds' (float).
        figsize (tuple): Size of the plot.
        cmap (Colormap): A matplotlib colormap.
        gpd_url (str) : url to shapefile for USA map plotting.
    """
    # Load shapefile (ensure this is extracted or accessible in your environment)
    counties = gpd.read_file(gpd_url)

    # Filter out non-continental states
    outside_states = ['Puerto Rico', 'Alaska', 'Hawaii']
    counties = counties[~counties['STATE_NAME'].isin(outside_states)]

    # Merge with preds
    counties['GEOID'] = counties['GEOID'].astype(int)
    df['fips'] = df['fips'].astype(int)
    counties = pd.merge(counties,
                        df[['fips', col]].rename(columns={'fips': 'GEOID'}),
                        on='GEOID',
                        how='left')

    # Set up color normalization
    vmin, vmax = counties[col].min(), counties[col].max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    counties.plot(column=col, cmap=cmap, linewidth=0.5, ax=ax, edgecolor='gray', norm=norm)

    # Colorbar (legend)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # Required for older matplotlib
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Predicted Value", fontsize=12)

    # Show
    ax.set_title("Predictions Across US Counties", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------------

def custom_objective(y_true, y_pred, thr_outages=100) :
    """
    Custom objective function (variant of asymmetric RMSE).

    2 differences :
    - Under-penalize over-estimations some hours ahead of peaks by 100%. These periods are caracterized with target finishes by '.01' instead of being an integer.
    - Over-penalize under-estimations errors on peaks (outages > thr) by 100%.
    """

    # Compute the difference between predictions and true values
    gradient = y_pred - y_true
    
    # ------------------------------------------------
    # Underpenalize when model forecast peaks just some hours too early

    # Get boolean mask of floats ending with .01, which caracterize periods just before outages peaks
    index_just_before_peaks = np.isclose(y_true % 1, 0.01) # (y_true % 1 == 0.01) can fail due to precision errors
    index_under_estimations, index_over_estimations = (gradient < 0), (gradient > 0)

    # Under-penalize over-estimations on these periods
    gradient = np.where(index_just_before_peaks & index_over_estimations, gradient/2, gradient)

    #print(len(gradient[index_just_before_peaks & index_under_estimations]), np.mean(gradient[index_just_before_peaks & index_under_estimations]))
    # ------------------------------------------------
    # Over-penalize under-estimations on peaks (outages > thr) by 100%.
    
    index_peaks = (y_true >= thr_outages)

    # Over-penalize under-estimations on peaks
    gradient = np.where(index_peaks & index_under_estimations, 2*gradient, gradient)

    # --------------------------------------------
                
    # Use a constant hessian for simplicity (or implement custom second-order logic if needed)
    hessian = np.ones_like(gradient)
        
    # Return
    return gradient, hessian

def custom_score_function(y_true, y_pred) :
    gradient, hessian = custom_objective(y_true, y_pred)
    return np.sqrt(np.mean(gradient**2))

def custom_metric(y_true, y_pred):
    return 'custom', custom_score_function(y_true, y_pred), False # False -> lower metric values are better

# -----------------------------------------------------------------------------------------------------

def custom_peak_Fbeta_score(df,
                            target_col,
                            pred_col,
                            thr_outages=100,
                            nhours_gap=4,
                            nhours_window=8,
                            beta=2,
                            display_result=False,
                           ) :

    """
    Custom metrics relevant to rare events (major outages outbreaks).
    -> Compute Precision, Recall, and Fβ-score for predicting upcoming outage outbreaks.

    The function evaluates if predictions correctly identify major outage outbreaks 
    happening between `nhours_gap` and `nhours_gap + nhours_window` hours ahead.

    An outbreak is defined as a maximum of ≥100 outages and ≥0.1% of the county's population during the future window.
    A prediction "finds" an outbreak if it covers at least 50% of the true value AND 100 outages.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least 'fips', 'outages', 'Population', and the specified `target_col` and `pred_col`.
    target_col : str
        Name of the true outages column.
    pred_col : str
        Name of the predicted outages column.
    thr_outages : float
        Minimal number of outages to surpass for an event to be considered a true outbreak.
    nhours_gap : int, default=4
        Minimum number of hours before checking for an outbreak.
    nhours_window : int, default=8
        Size of the future window to monitor for an outbreak.
    beta : float, optional (default=2)
        β parameter of the Fβ-score:
        - β > 1 favors Recall (important if missing an outbreak is very costly),
        - β < 1 favors Precision (important if false alarms are very costly),
        - β = 1 corresponds to the standard F1-score.
    display_result : bool (default False)
        Display and print results.

    Returns
    -------
    precision : float
        Proportion of predicted outbreaks that were correct.
    recall : float
        Proportion of true outbreaks that were successfully predicted.
    fbeta_score : float
        Weighted Fβ-score balancing precision and recall according to `beta`.
    """

    # Find max outages in the future from hour H, in the periods [H + gap, H + gap + window]
    n1, n2 = nhours_gap, nhours_gap + nhours_window
    df[f"max_outages_outbreak_from_{n1}_to_{n2}_hours"] = df.groupby(df['fips'].astype(int))["outages"].rolling(window=nhours_window, min_periods=1).max().values
    df[f"max_outages_outbreak_from_{n1}_to_{n2}_hours"] = df[f"max_outages_outbreak_from_{n1}_to_{n2}_hours"].shift(-nhours_gap-1)

    # Find if an outages outbreak is about to happen (more than 'thr_outages' outages AND 0.1% of the county population for period [H + gap, H + gap + window])
    df["incoming_outages_outbreak"] = (
        (df[f"max_outages_outbreak_from_{n1}_to_{n2}_hours"] >= thr_outages) &
        (df[f"max_outages_outbreak_from_{n1}_to_{n2}_hours"] >= 0.001 * df["Population"])
    )

    # We consider that we find the outages if we predict at least half of the outages and more than 'thr_outages' outages
    df["found_outbreak"] = (((df[pred_col] / df[target_col]) >= 0.5) & (df[pred_col]>=thr_outages))

    # Identify rows where a true outbreak is incoming, and which one we found
    mask_incoming_outbreaks = (df["incoming_outages_outbreak"] == 1)
    mask_found_outbreaks    = (df["found_outbreak"] == 1)

    # If there are no true outbreaks, avoid division by zero
    if mask_incoming_outbreaks.sum() == 0:
        return 0.0, 0.0, 0.0

    # Compute some metrics (F1, precision and recall)
    true_positive      = (mask_incoming_outbreaks & mask_found_outbreaks).sum()
    predicted_positive = mask_found_outbreaks.sum()
    actual_positive    = mask_incoming_outbreaks.sum()

    precision = true_positive / predicted_positive if predicted_positive else 0.0 # Precision (important if false alarms are very costly),
    recall = true_positive / actual_positive # Recall (important if missing an outbreak is very costly),
    fbeta_score = (1+beta**2) * precision * recall / ((beta**2)*precision + recall) if (precision + recall) else 0.0

    # -----------------------------------------------------------------
    # Display information
    if display_result :
        
        # Round
        precision = round(100*precision, 1)
        recall = round(100*recall, 1)
        fbeta_score = round(100*fbeta_score, 1)
        
        # Print info
        print(f"PRECISION: {precision}%.")
        print(f"-> Among predicted outage outbreaks, {precision}% are correct.")
        print(f"   -> Important metric if false alarms are costly.")
        
        print()
        
        print(f"RECALL: {recall}%.")
        print(f"-> We find {recall}% of true outage outbreaks (4 to 12 hours ahead).")
        print(f"   -> Important metric if missing an outbreak is costly.")
        
        print()
        
        print(f"Fβ-score: {fbeta_score}%.")
        print(f"-> β=2, favors Recall.")

    # Return
    return precision, recall, fbeta_score

# -----------------------------------------------------------------------------------------------------

def plot_fbeta_curve(df,
                     target_col="outages_in_6hours",
                     pred_col="preds",
                     THR_range=range(100, 10000, 200),
                    ):
    
    """
    This function plots how the model's performance (precision, recall, and Fβ-score)
    varies depending on different thresholds for detecting significant outage peaks.
    It is useful for analyzing the trade-off between detecting more events (recall)
    and maintaining accuracy (precision) at different severity levels.

    Args:
        df (pd.DataFrame): 
            DataFrame containing the true targets and model predictions.
        target_col (str, optional): 
            Name of the column with true labels. Defaults to "outages_in_6hours".
        pred_col (str, optional): 
            Name of the column with model predictions. Defaults to "preds".
        THR_range (iterable, optional): 
            Range of outage thresholds to test. Each threshold defines what constitutes
            a "peak" or significant event. Defaults to range(500, 10000, 100).
    """

    # Initialize lists
    F_scores = []
    Recalls  = []
    Precisions = []

    # Iterate over thr
    for thr_outages in THR_range :
        precision, recall, fbeta_score = custom_peak_Fbeta_score(df,
                                                                 target_col="outages_in_6hours",
                                                                 pred_col="preds",
                                                                 thr_outages=thr_outages,
                                                                 beta=2)
        Precisions.append(precision)
        Recalls.append(recall)
        F_scores.append(fbeta_score)
        
    # Plot a graph
    plt.figure(figsize=(10, 4))
    plt.plot(THR_range, Precisions, label = "Precision", lw=2, c='C1')
    plt.plot(THR_range, Recalls, label = "Recall", lw=2, c='C3')
    plt.plot(THR_range, F_scores, label = "Fβ-score", lw=3, c='C0')
    plt.xlabel("Threshold on Outages")
    plt.ylabel("Score (%)")
    plt.title("Model Performance vs Outages Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------

# Optimized data types for df "df_hourly_outages". Helps save memory when we read csv with pandas.
dtypes = {'fips': 'int32',
         'date': 'object',
         'month': 'int8',
         'week': 'float32',
         'dayofweek': 'int8',
         'hour': 'int8',
         'outages': 'float32',
         'outages_outbreak': 'int32',
         'flag_more_than_50_outages': 'int8',
         'Proportion_of_fips_within_300km_with_more_than_50_outages': 'float32',
         'Proportion_of_fips_within_600km_with_more_than_50_outages': 'float32',
         'max_Proportion_of_fips_within_300km_with_more_than_50_outages_over_last_24hours': 'float32',
         'max_Proportion_of_fips_within_600km_with_more_than_50_outages_over_last_24hours': 'float32',
         'Temperature_1day_before': 'float32',
         'MaxWindSpeed_1day_before': 'float32',
         'Max_Temperature_over_3last_days': 'float32',
         'Mean_Temperature_over_3last_days': 'float32',
         'Max_MaxWindSpeed_over_3last_days': 'float32',
         'Mean_MaxWindSpeed_over_3last_days': 'float32',
         'Severe_events_count': 'float32',
         'LandArea': 'float32',
         'WaterArea': 'float32',
         'latitude': 'float32',
         'longitude': 'float32',
         'Age': 'float32',
         'Population': 'float32',
         'state': 'object',
         'max_state_fips_with_more_than_50_outages_over_last_6hours': 'float32',
         'proportion_state_fips_with_more_than_50_outages_over_last_6hours': 'float32',
         'max_state_fips_with_more_than_50_outages_over_last_24hours': 'float32',
         'proportion_state_fips_with_more_than_50_outages_over_last_24hours': 'float32',
         'SUM_BY_STATE_Severe_events_count_sum_by_state_last_24hours': 'float32',
         'SUM_BY_STATE_WORDS_wind_and_storm_sum_by_state_last_24hours': 'float32',
         'SUM_BY_STATE_WORDS_downed_trees_sum_by_state_last_24hours': 'float32',
         'SUM_BY_STATE_WORDS_heavy_damage_sum_by_state_last_24hours': 'float32',
         'SUM_BY_STATE_WORDS_wires_sum_by_state_last_24hours': 'float32',
         'SUM_BY_STATE_state_fips_with_more_than_50_outages_last_24hours': 'float32',
         'projected_distance_to_centroid_center_in_3hours': 'float32',
         'vector_weights': 'float32',
         'vector_weights_diff': 'float32',
         'vector_cluster_size': 'float32',
         'vector_cluster_size_diff': 'float32',
         'projected_fips_impacted_in_3hours': 'float32',
         'projected_impact_in_3hours': 'float32',
         'projected_distance_to_centroid_center_in_6hours': 'float32',
         'projected_fips_impacted_in_6hours': 'float32',
         'projected_impact_in_6hours': 'float32',
         'outages_sum_among_1st_neighbours': 'float32',
         'outages_outbreak_sum_among_1st_neighbours': 'float32',
         'Severe_events_count_sum_among_1st_neighbours': 'float32',
         'Temperature_1day_before_std_among_1st_neighbours': 'float32',
         'MAGNITUDE_sum_among_1st_neighbours': 'float32',
         'Population_sum_among_1st_neighbours': 'float32',
         'outages_sum_among_2nd_neighbours': 'float32',
         'outages_outbreak_sum_among_2nd_neighbours': 'float32',
         'Severe_events_count_sum_among_2nd_neighbours': 'float32',
         'Temperature_1day_before_std_among_2nd_neighbours': 'float32',
         'MAGNITUDE_sum_among_2nd_neighbours': 'float32',
         'Population_sum_among_2nd_neighbours': 'float32',
         'outages_sum_among_3rd_neighbours': 'float32',
         'outages_outbreak_sum_among_3rd_neighbours': 'float32',
         'Severe_events_count_sum_among_3rd_neighbours': 'float32',
         'Temperature_1day_before_std_among_3rd_neighbours': 'float32',
         'Population_sum_among_3rd_neighbours': 'float32',
         'outages_sum_among_4th_neighbours': 'float32',
         'outages_outbreak_sum_among_4th_neighbours': 'float32',
         'Severe_events_count_sum_among_4th_neighbours': 'float32',
         'Temperature_1day_before_std_among_4th_neighbours': 'float32',
         'Population_sum_among_4th_neighbours': 'float32',
         'outages_sum_among_ALL_neighbours': 'float32',
         'outages_outbreak_sum_among_ALL_neighbours': 'float32',
         'Severe_events_count_sum_among_ALL_neighbours': 'float32',
         'Population_sum_among_ALL_neighbours': 'float32',
         'WORDS_wind_and_storm_sum_among_ALL_neighbours': 'float32',
         'WORDS_downed_trees_sum_among_ALL_neighbours': 'float32',
         'WORDS_heavy_damage_sum_among_ALL_neighbours': 'float32',
         'WORDS_wires_sum_among_ALL_neighbours': 'float32',
         'WORDS_heat_and_fire_sum_among_ALL_neighbours': 'float32',
         'Sum_of_WORDS_wind_and_storm_sum_among_ALL_neighbours_over_last_6hours': 'float32',
         'Sum_of_WORDS_downed_trees_sum_among_ALL_neighbours_over_last_6hours': 'float32',
         'Sum_of_WORDS_wires_sum_among_ALL_neighbours_over_last_6hours': 'float32',
         'Sum_of_Severe_events_count_sum_among_1st_neighbours_over_last_6hours': 'float32',
         'Sum_of_Severe_events_count_sum_among_2nd_neighbours_over_last_6hours': 'float32',
         'Sum_of_Severe_events_count_sum_among_ALL_neighbours_over_last_6hours': 'float32',
         'Sum_of_WORDS_wind_and_storm_sum_among_ALL_neighbours_over_last_24hours': 'float32',
         'Sum_of_WORDS_downed_trees_sum_among_ALL_neighbours_over_last_24hours': 'float32',
         'Sum_of_WORDS_wires_sum_among_ALL_neighbours_over_last_24hours': 'float32',
         'Sum_of_Severe_events_count_sum_among_1st_neighbours_over_last_24hours': 'float32',
         'Sum_of_Severe_events_count_sum_among_2nd_neighbours_over_last_24hours': 'float32',
         'Sum_of_Severe_events_count_sum_among_ALL_neighbours_over_last_24hours': 'float32',
         'Max_of_outages_over_last_12hours': 'float32',
         'Max_of_outages_sum_among_1st_neighbours_over_last_12hours': 'float32',
         'Max_of_outages_sum_among_2nd_neighbours_over_last_12hours': 'float32',
         'Max_of_outages_sum_among_ALL_neighbours_over_last_12hours': 'float32',
         'Max_of_outages_over_last_36hours': 'float32',
         'Max_of_outages_sum_among_1st_neighbours_over_last_36hours': 'float32',
         'Max_of_outages_sum_among_2nd_neighbours_over_last_36hours': 'float32',
         'Max_of_outages_sum_among_ALL_neighbours_over_last_36hours': 'float32',
         'Max_of_MAGNITUDE_sum_among_1st_neighbours_over_last_12hours': 'float32',
         'Max_of_MAGNITUDE_sum_among_2nd_neighbours_over_last_12hours': 'float32',
         'DIFF_outages': 'float32',
         'DIFF_outages_from_6hours_ago': 'float32',
         'Biggest_raise_of_outages_over_last_6h': 'float32',
         'DIFF_outages_from_24hours_ago': 'float32',
         'Biggest_raise_of_outages_over_last_24h': 'float32',
         'DIFF_outages_outbreak': 'float32',
         'DIFF_outages_outbreak_from_6hours_ago': 'float32',
         'Biggest_raise_of_outages_outbreak_over_last_6h': 'float32',
         'DIFF_outages_outbreak_from_24hours_ago': 'float32',
         'Biggest_raise_of_outages_outbreak_over_last_24h': 'float32',
         'outages_in_6hours': 'float32',
         'outages_outbreak_in_6hours': 'float32'}

# -----------------------------------------------------------------------------------------------------
# List of fips for different regions (nort, west, east, south)

north_east_fips = [9001, 9003, 9005, 9007, 9009, 9011, 9013, 9015, 10001, 10003, 10005,
                   23001, 23003, 23005, 23007, 23009, 23011, 23013, 23015, 23017, 23019, 23021, 23023, 23025, 23027, 23029, 23031,
                   24001, 24003, 24005, 24009, 24011, 24013, 24015, 24017, 24019, 24021, 24023, 24025, 24027, 24029, 24031, 24033,
                   24035, 24037, 24039, 24041, 24043, 24045, 24047, 24510, 25001, 25003, 25005, 25007, 25009, 25011, 25013, 25015,
                   25017, 25019, 25021, 25023, 25025, 25027, 33001, 33003, 33005, 33007, 33009, 33011, 33013, 33015, 33017, 33019,
                   34001, 34003, 34005, 34007, 34009, 34011, 34013, 34015, 34017, 34019, 34021, 34023, 34025, 34027, 34029, 34031,
                   34033, 34035, 34037, 34039, 34041, 36001, 36003, 36005, 36007, 36009, 36011, 36013, 36015, 36017, 36019, 36021,
                   36023, 36025, 36027, 36029, 36031, 36033, 36035, 36037, 36039, 36041, 36043, 36045, 36047, 36049, 36051, 36053,
                   36055, 36057, 36059, 36061, 36063, 36065, 36067, 36069, 36071, 36073, 36075, 36077, 36079, 36081, 36083, 36085,
                   36087, 36089, 36091, 36093, 36095, 36097, 36099, 36101, 36103, 36105, 36107, 36109, 36111, 36113, 36115, 36117,
                   36119, 36121, 36123, 42001, 42003, 42005, 42007, 42009, 42011, 42013, 42015, 42017, 42019, 42021, 42023, 42025,
                   42027, 42029, 42031, 42033, 42035, 42037, 42039, 42041, 42043, 42045, 42047, 42049, 42051, 42053, 42055, 42057,
                   42059, 42061, 42063, 42065, 42067, 42069, 42071, 42073, 42075, 42077, 42079, 42081, 42083, 42085, 42087, 42089,
                   42091, 42093, 42095, 42097, 42099, 42101, 42103, 42105, 42107, 42109, 42111, 42113, 42115, 42117, 42119, 42121,
                   42123, 42125, 42127, 42129, 42131, 42133, 44001, 44003, 44005, 44007, 44009, 50001, 50003, 50005, 50007, 50009,
                   50011, 50013, 50015, 50017, 50019, 50021, 50023, 50025, 50027]
                    
west_fips = [2020, 2068, 2090, 2122, 2170, 2240, 2290, 4001, 4003, 4005, 4007, 4012, 4013, 4015, 4017, 4019, 4021, 4023, 4025,
             4027, 6001, 6003, 6005, 6007, 6009, 6011, 6013, 6015, 6017, 6019, 6021, 6023, 6025, 6027, 6029, 6031, 6033, 6035,
             6037, 6039, 6041, 6043, 6045, 6047, 6049, 6051, 6053, 6055, 6057, 6059, 6061, 6063, 6065, 6067, 6069, 6071, 6073,
             6075, 6077, 6079, 6081, 6083, 6085, 6087, 6089, 6091, 6093, 6095, 6097, 6099, 6101, 6103, 6105, 6107, 6109, 6111,
             6113, 6115, 8001, 8003, 8005, 8007, 8009, 8013, 8014, 8015, 8017, 8019, 8021, 8023, 8025, 8027, 8029, 8031, 8035,
             8037, 8039, 8041, 8043, 8045, 8047, 8049, 8051, 8053, 8059, 8063, 8065, 8067, 8069, 8073, 8075, 8077, 8079, 8081,
             8085, 8087, 8089, 8091, 8093, 8097, 8101, 8103, 8105, 8107, 8109, 8111, 8117, 8119, 8121, 8123, 15001, 15003, 15007,
             15009, 16001, 16003, 16005, 16007, 16009, 16011, 16013, 16015, 16017, 16019, 16021, 16023, 16025, 16027, 16029, 16031,
             16033, 16035, 16039, 16041, 16043, 16045, 16047, 16049, 16051, 16053, 16055, 16057, 16059, 16061, 16063, 16065, 16067,
             16069, 16071, 16073, 16075, 16077, 16079, 16081, 16083, 16085, 16087, 30001, 30003, 30005, 30007, 30009, 30013, 30015,
             30017, 30019, 30021, 30023, 30025, 30027, 30029, 30031, 30037, 30039, 30041, 30043, 30045, 30047, 30049, 30051, 30053,
             30057, 30059, 30061, 30063, 30065, 30067, 30071, 30073, 30075, 30077, 30079, 30081, 30083, 30085, 30087, 30089, 30091,
             30093, 30095, 30097, 30099, 30105, 30107, 30109, 30111, 32001, 32003, 32005, 32007, 32009, 32011, 32013, 32015, 32019,
             32021, 32023, 32027, 32029, 32031, 32510, 35001, 35003, 35005, 35009, 35011, 35013, 35015, 35017, 35019, 35023, 35025,
             35027, 35029, 35031, 35035, 35037, 35039, 35041, 35043, 35045, 35047, 35049, 35051, 35053, 35057, 35059, 35061, 41001,
             41003, 41005, 41007, 41009, 41011, 41013, 41015, 41017, 41019, 41021, 41023, 41025, 41027, 41029, 41031, 41033, 41035,
             41037, 41039, 41041, 41043, 41045, 41047, 41049, 41051, 41053, 41055, 41057, 41059, 41061, 41063, 41065, 41067, 41071,
             49001, 49003, 49005, 49007, 49011, 49013, 49015, 49017, 49019, 49021, 49023, 49027, 49029, 49031, 49033, 49035, 49037,
             49039, 49041, 49043, 49045, 49047, 49049, 49051, 49053, 49057, 53001, 53003, 53005, 53007, 53009, 53011, 53013, 53015,
             53017, 53019, 53021, 53023, 53025, 53029, 53031, 53033, 53035, 53037, 53041, 53043, 53045, 53051, 53053, 53055, 53057,
             53061, 53063, 53065, 53067, 53071, 53073, 53075, 53077, 56001, 56003, 56005, 56007, 56009, 56011, 56013, 56017, 56019,
             56021, 56023, 56025, 56029, 56031, 56033, 56035, 56037, 56039, 56041, 56043, 56045]

south_fips = [1001, 1003, 1005, 1007, 1009, 1011, 1013, 1015, 1017, 1019, 1021, 1023, 1025, 1027, 1029, 1031, 1035, 1037, 1039, 1041, 1043, 1045, 1047, 1051, 1053, 1055, 1057, 1059, 1061, 1063, 1065, 1067, 1069, 1071, 1073, 1075, 1077, 1079, 1081, 1083, 1085, 1087, 1089, 1091, 1093, 1095, 1097, 1099, 1101, 1103, 1105, 1107, 1109, 1111, 1113, 1115, 1117, 1119, 1121, 1123, 1125, 1127, 1129, 1131, 1133, 5001, 5003, 5005, 5007, 5009, 5011, 5013, 5015, 5017, 5019, 5021, 5023, 5025, 5027, 5029, 5031, 5033, 5035, 5037, 5039, 5041, 5043, 5045, 5047, 5049, 5051, 5053, 5055, 5057, 5059, 5061, 5063, 5065, 5067, 5069, 5071, 5073, 5075, 5077, 5079, 5081, 5083, 5085, 5087, 5089, 5091, 5093, 5095, 5097, 5099, 5101, 5103, 5105, 5107, 5109, 5111, 5113, 5115, 5117, 5119, 5121, 5123, 5125, 5127, 5129, 5131, 5133, 5135, 5137, 5139, 5141, 5143, 5145, 5147, 5149, 20001, 20003, 20005, 20007, 20009, 20011, 20013, 20015, 20017, 20021, 20023, 20027, 20029, 20031, 20035, 20037, 20039, 20041, 20043, 20045, 20047, 20049, 20051, 20055, 20059, 20061, 20063, 20065, 20069, 20071, 20073, 20075, 20077, 20079, 20083, 20085, 20087, 20089, 20091, 20093, 20095, 20099, 20103, 20105, 20107, 20109, 20111, 20113, 20115, 20117, 20121, 20123, 20125, 20127, 20129, 20131, 20133, 20135, 20137, 20139, 20141, 20143, 20145, 20147, 20149, 20151, 20153, 20155, 20157, 20159, 20161, 20163, 20165, 20167, 20169, 20171, 20173, 20175, 20177, 20179, 20181, 20183, 20185, 20191, 20193, 20195, 20197, 20199, 20201, 20203, 20205, 20207, 20209, 21001, 21003, 21005, 21007, 21009, 21011, 21013, 21015, 21017, 21019, 21021, 21023, 21025, 21027, 21029, 21031, 21033, 21035, 21037, 21039, 21041, 21043, 21045, 21047, 21049, 21051, 21053, 21055, 21057, 21059, 21061, 21063, 21065, 21067, 21069, 21071, 21073, 21075, 21077, 21079, 21081, 21083, 21085, 21087, 21089, 21091, 21093, 21095, 21097, 21099, 21101, 21103, 21105, 21107, 21109, 21111, 21113, 21115, 21117, 21119, 21121, 21123, 21125, 21127, 21129, 21131, 21133, 21135, 21137, 21139, 21141, 21143, 21145, 21147, 21149, 21151, 21153, 21155, 21157, 21159, 21161, 21163, 21165, 21167, 21169, 21171, 21173, 21175, 21177, 21179, 21181, 21183, 21185, 21187, 21189, 21191, 21193, 21195, 21197, 21199, 21201, 21203, 21205, 21207, 21209, 21211, 21213, 21215, 21217, 21219, 21221, 21223, 21225, 21227, 21229, 21231, 21233, 21235, 21237, 21239, 22001, 22003, 22005, 22007, 22009, 22011, 22013, 22015, 22017, 22019, 22021, 22023, 22025, 22027, 22029, 22031, 22033, 22035, 22037, 22039, 22041, 22043, 22045, 22047, 22049, 22051, 22053, 22055, 22057, 22059, 22061, 22063, 22065, 22067, 22069, 22071, 22073, 22075, 22077, 22079, 22081, 22083, 22085, 22087, 22089, 22091, 22093, 22095, 22097, 22099, 22101, 22103, 22105, 22107, 22109, 22111, 22113, 22115, 22117, 22119, 22121, 22123, 22125, 22127, 28001, 28003, 28005, 28007, 28009, 28011, 28013, 28015, 28017, 28019, 28021, 28023, 28025, 28027, 28029, 28031, 28033, 28035, 28037, 28039, 28041, 28043, 28045, 28047, 28049, 28051, 28053, 28057, 28059, 28061, 28063, 28065, 28067, 28069, 28071, 28073, 28075, 28077, 28079, 28083, 28085, 28087, 28089, 28091, 28093, 28095, 28097, 28099, 28101, 28103, 28105, 28107, 28109, 28111, 28113, 28115, 28119, 28121, 28123, 28125, 28127, 28129, 28131, 28133, 28135, 28137, 28139, 28143, 28145, 28147, 28149, 28151, 28153, 28155, 28157, 28159, 28161, 28163, 29001, 29003, 29005, 29007, 29009, 29011, 29013, 29015, 29017, 29019, 29021, 29023, 29025, 29027, 29029, 29031, 29033, 29035, 29037, 29039, 29041, 29043, 29045, 29047, 29049, 29051, 29053, 29055, 29057, 29059, 29061, 29063, 29065, 29067, 29069, 29071, 29073, 29075, 29077, 29079, 29081, 29083, 29085, 29087, 29089, 29091, 29093, 29095, 29097, 29099, 29101, 29103, 29105, 29107, 29109, 29111, 29113, 29115, 29117, 29119, 29121, 29123, 29125, 29127, 29129, 29131, 29133, 29135, 29137, 29139, 29141, 29143, 29145, 29147, 29149, 29151, 29153, 29155, 29157, 29159, 29161, 29163, 29165, 29167, 29169, 29171, 29173, 29175, 29177, 29179, 29181, 29183, 29185, 29186, 29187, 29195, 29197, 29199, 29201, 29203, 29205, 29207, 29209, 29211, 29213, 29215, 29217, 29219, 29221, 29223, 29225, 29227, 29229, 29510, 40001, 40003, 40005, 40007, 40009, 40011, 40013, 40015, 40017, 40019, 40021, 40023, 40025, 40027, 40029, 40031, 40033, 40035, 40037, 40039, 40041, 40043, 40045, 40047, 40049, 40051, 40053, 40055, 40057, 40059, 40061, 40063, 40065, 40067, 40069, 40071, 40073, 40075, 40077, 40079, 40081, 40083, 40085, 40087, 40089, 40091, 40093, 40095, 40097, 40099, 40101, 40103, 40105, 40107, 40109, 40111, 40113, 40115, 40117, 40119, 40121, 40123, 40125, 40127, 40129, 40131, 40133, 40135, 40137, 40139, 40141, 40143, 40145, 40147, 40149, 40151, 40153, 47001, 47003, 47007, 47009, 47011, 47013, 47015, 47021, 47023, 47025, 47027, 47029, 47031, 47033, 47035, 47037, 47041, 47045, 47047, 47049, 47051, 47053, 47055, 47057, 47059, 47061, 47063, 47065, 47067, 47069, 47073, 47075, 47077, 47079, 47081, 47089, 47093, 47095, 47097, 47099, 47101, 47103, 47105, 47107, 47111, 47113, 47115, 47117, 47119, 47121, 47123, 47125, 47127, 47129, 47131, 47133, 47137, 47139, 47141, 47143, 47145, 47147, 47149, 47151, 47153, 47155, 47157, 47159, 47161, 47163, 47165, 47167, 47169, 47171, 47173, 47175, 47177, 47179, 47181, 47183, 47185, 47187, 47189, 48001, 48003, 48005, 48007, 48009, 48011, 48013, 48015, 48017, 48019, 48021, 48023, 48025, 48027, 48029, 48031, 48033, 48035, 48037, 48039, 48041, 48043, 48045, 48047, 48049, 48051, 48053, 48055, 48057, 48059, 48061, 48063, 48065, 48067, 48069, 48071, 48073, 48075, 48077, 48079, 48081, 48083, 48085, 48087, 48089, 48091, 48093, 48095, 48097, 48099, 48101, 48103, 48105, 48107, 48109, 48111, 48113, 48115, 48117, 48119, 48121, 48123, 48125, 48127, 48129, 48131, 48133, 48135, 48137, 48139, 48141, 48143, 48145, 48147, 48149, 48151, 48153, 48155, 48157, 48159, 48161, 48163, 48165, 48167, 48169, 48171, 48173, 48175, 48177, 48179, 48181, 48183, 48185, 48187, 48189, 48191, 48193, 48195, 48197, 48199, 48201, 48203, 48205, 48207, 48209, 48211, 48213, 48215, 48217, 48219, 48221, 48223, 48225, 48227, 48229, 48231, 48233, 48235, 48237, 48239, 48241, 48243, 48245, 48247, 48249, 48251, 48253, 48255, 48257, 48259, 48261, 48263, 48265, 48267, 48269, 48271, 48273, 48275, 48277, 48279, 48281, 48283, 48285, 48287, 48289, 48291, 48293, 48295, 48297, 48299, 48301, 48303, 48305, 48307, 48309, 48311, 48313, 48315, 48317, 48319, 48321, 48323, 48325, 48327, 48329, 48331, 48333, 48335, 48337, 48339, 48341, 48343, 48345, 48347, 48349, 48353, 48355, 48357, 48359, 48361, 48363, 48365, 48367, 48369, 48371, 48373, 48375, 48377, 48379, 48381, 48383, 48385, 48387, 48389, 48391, 48393, 48395, 48397, 48399, 48401, 48407, 48409, 48411, 48413, 48415, 48417, 48419, 48421, 48423, 48425, 48427, 48429, 48431, 48433, 48435, 48437, 48439, 48441, 48443, 48445, 48447, 48449, 48451, 48453, 48455, 48457, 48459, 48461, 48463, 48465, 48467, 48469, 48471, 48473, 48475, 48477, 48479, 48481, 48483, 48485, 48487, 48489, 48491, 48493, 48495, 48497, 48499, 48501, 48503, 48505, 48507]

central_fips = [17001, 17003, 17005, 17007, 17009, 17011, 17013, 17015, 17017, 17019, 17021, 17023, 17025, 17027, 17029, 17031, 17033, 17035, 17037, 17039, 17041, 17043, 17045, 17047, 17049, 17051, 17053, 17055, 17057, 17059, 17061, 17063, 17065, 17067, 17069, 17071, 17073, 17075, 17077, 17079, 17081, 17083, 17085, 17087, 17089, 17091, 17093, 17095, 17097, 17099, 17101, 17103, 17105, 17107, 17109, 17111, 17113, 17115, 17117, 17119, 17121, 17123, 17125, 17127, 17129, 17131, 17133, 17135, 17137, 17139, 17141, 17143, 17145, 17147, 17149, 17151, 17153, 17155, 17157, 17159, 17161, 17163, 17165, 17167, 17169, 17171, 17173, 17175, 17177, 17179, 17181, 17183, 17185, 17187, 17189, 17191, 17193, 17195, 17197, 17199, 17201, 17203, 18001, 18003, 18005, 18007, 18009, 18011, 18013, 18015, 18017, 18019, 18021, 18023, 18025, 18027, 18029, 18031, 18033, 18035, 18039, 18041, 18043, 18045, 18047, 18049, 18051, 18053, 18055, 18057, 18059, 18061, 18063, 18065, 18067, 18069, 18071, 18073, 18075, 18077, 18079, 18081, 18083, 18085, 18087, 18089, 18091, 18093, 18095, 18097, 18099, 18101, 18103, 18105, 18107, 18109, 18111, 18113, 18115, 18117, 18119, 18121, 18125, 18127, 18129, 18131, 18133, 18135, 18137, 18139, 18141, 18143, 18145, 18147, 18149, 18151, 18153, 18155, 18157, 18159, 18161, 18163, 18165, 18167, 18169, 18171, 18173, 18175, 18177, 18179, 18181, 18183, 19001, 19003, 19005, 19007, 19009, 19011, 19013, 19015, 19017, 19019, 19021, 19023, 19025, 19027, 19029, 19031, 19033, 19035, 19037, 19039, 19041, 19043, 19045, 19047, 19049, 19051, 19053, 19055, 19057, 19059, 19061, 19063, 19065, 19067, 19069, 19071, 19073, 19075, 19077, 19079, 19081, 19083, 19085, 19087, 19089, 19091, 19093, 19095, 19097, 19099, 19101, 19103, 19105, 19107, 19109, 19111, 19113, 19115, 19117, 19119, 19121, 19123, 19125, 19127, 19129, 19131, 19133, 19135, 19137, 19139, 19141, 19143, 19145, 19147, 19149, 19151, 19153, 19155, 19157, 19159, 19161, 19163, 19165, 19167, 19169, 19171, 19173, 19175, 19177, 19179, 19181, 19183, 19185, 19187, 19189, 19191, 19193, 19195, 19197, 27001, 27003, 27005, 27007, 27009, 27011, 27013, 27015, 27017, 27019, 27021, 27023, 27025, 27027, 27035, 27037, 27039, 27041, 27043, 27045, 27047, 27049, 27051, 27053, 27055, 27057, 27059, 27061, 27065, 27067, 27069, 27071, 27073, 27075, 27079, 27081, 27083, 27085, 27087, 27089, 27093, 27095, 27097, 27099, 27101, 27103, 27107, 27109, 27111, 27115, 27117, 27119, 27121, 27123, 27125, 27127, 27129, 27131, 27133, 27135, 27137, 27139, 27141, 27143, 27145, 27147, 27149, 27151, 27153, 27155, 27157, 27159, 27161, 27163, 27167, 27169, 27171, 27173, 31003, 31015, 31017, 31019, 31021, 31023, 31025, 31027, 31031, 31037, 31043, 31045, 31047, 31049, 31053, 31055, 31059, 31065, 31067, 31069, 31079, 31081, 31083, 31089, 31093, 31095, 31097, 31101, 31107, 31109, 31111, 31119, 31123, 31127, 31131, 31133, 31135, 31139, 31145, 31147, 31149, 31151, 31153, 31155, 31157, 31159, 31161, 31163, 31165, 31169, 31173, 31177, 31185, 38001, 38003, 38005, 38009, 38011, 38013, 38015, 38017, 38019, 38021, 38023, 38025, 38027, 38029, 38031, 38033, 38035, 38037, 38039, 38041, 38043, 38045, 38047, 38049, 38051, 38053, 38055, 38057, 38059, 38061, 38063, 38065, 38067, 38069, 38071, 38073, 38075, 38077, 38079, 38081, 38083, 38085, 38089, 38091, 38093, 38095, 38097, 38099, 38101, 38103, 38105, 46003, 46005, 46007, 46009, 46011, 46013, 46015, 46017, 46019, 46021, 46023, 46025, 46027, 46029, 46031, 46033, 46035, 46037, 46039, 46043, 46045, 46047, 46049, 46051, 46053, 46055, 46057, 46059, 46061, 46063, 46065, 46067, 46069, 46071, 46073, 46075, 46077, 46079, 46081, 46083, 46085, 46087, 46089, 46091, 46093, 46095, 46097, 46099, 46101, 46102, 46103, 46105, 46107, 46109, 46111, 46115, 46117, 46123, 46125, 46127, 46129, 46135, 46137, 55001, 55003, 55005, 55007, 55009, 55011, 55013, 55015, 55017, 55019, 55021, 55023, 55025, 55027, 55029, 55031, 55033, 55035, 55037, 55039, 55041, 55043, 55045, 55047, 55049, 55051, 55053, 55055, 55057, 55059, 55061, 55063, 55065, 55067, 55069, 55071, 55073, 55075, 55077, 55078, 55079, 55081, 55083, 55085, 55087, 55089, 55091, 55093, 55095, 55097, 55099, 55101, 55103, 55105, 55107, 55109, 55111, 55113, 55115, 55117, 55119, 55121, 55123, 55125, 55127, 55129, 55131, 55133, 55135, 55137, 55139, 55141]

# -----------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------
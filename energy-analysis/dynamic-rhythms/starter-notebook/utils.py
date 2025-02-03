import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests, ccf
import numpy as np
import plotly.express as px
from datetime import datetime
import geopandas as gpd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def plot_outages_on_map_us(start_time,
                           end_time,
                           data_directory = './data/eaglei_data'):
    """
    Plots the sum of customer outages on the map of the USA using Plotly.

    Parameters:
    start_time (str): The start time in ISO format (YYYY-MM-DDTHH:MM:SS).
    end_time (str): The end time in ISO format (YYYY-MM-DDTHH:MM:SS).
    data_directory (str, optional): The directory containing the CSV files. Default is './data/eaglei_data'.

    Raises:
    ValueError: If the year extracted from start_time is not between 2014 and 2023.
    FileNotFoundError: If the CSV file for the specified year does not exist in the data_directory.

    Returns:
    None
    """
    
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
    'Wyoming': 'WY', 'United States Virgin Islands': 'VI'
    }

    # Convert start_time and end_time to a datetime object and extract the year
    start_time = datetime.fromisoformat(start_time)
    end_time = datetime.fromisoformat(end_time)
    year = start_time.year
    
    # Check if the year is within the valid range
    valid_years = [str(y) for y in range(2014, 2024)]
    if str(year) not in valid_years:
        raise ValueError(f"Invalid year: {year}. Year must be between 2014 and 2023.")
    
    # Construct the file path
    file_name = f"eaglei_outages_{year}.csv"
    file_path = os.path.join(data_directory, file_name)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_name} does not exist in the directory {data_directory}.")
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    df['run_start_time'] = pd.to_datetime(df['run_start_time'])
    df.dropna(subset=['customers_out'],inplace=True)
    

    df['state'] = df['state'].map(state_abbreviations)
    # Filter data based on time range
    filtered_df = df[(df['run_start_time'] >= start_time) & (df['run_start_time'] <= end_time)]
    
    grouped = filtered_df.groupby('state').agg({
        'customers_out': 'sum'
    }).reset_index()
    grouped['customers_out'] = grouped['customers_out'] / 96
    title = "Power Outages by State"
    location_col = 'state'
    
    # Normalize colormap to the data range
    max_outages = grouped['customers_out'].max()
    min_outages = grouped['customers_out'].min()
    

    # Create choropleth map
    fig = px.choropleth(
        grouped,
        locations='state',
        locationmode='USA-states',
        color='customers_out',
        color_continuous_scale="OrRd",
        scope="usa",
        range_color=(min_outages, max_outages),  # Fix normalization
        title=f"{title} ({start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')})",
        labels={'customers_out': 'customers.day'}
    )
    
    # Add hover info to display total numbers
    fig.update_traces(
        hovertemplate='<b>%{location}</b><br>Total Outage: %{z}<extra></extra>'
    )
    
    fig.update_geos(showcoastlines=True, coastlinecolor="Black")
    
    fig.update_layout(
        geo=dict(
            lakecolor="lightblue",
            showlakes=True
        ),
        title_x=0.4,  # Center the title
        margin=dict(l=0, r=0, t=50, b=0),  # Minimize space between colorbar and map figure
        width=1000, height=600,  # Set figure size to (10, 6)
        title_font=dict(size=20, family='Arial, bold'),  # Set figure title font size and make it bold
        font=dict(size=14)  # Set default font size for the figure
    )
    
    fig.show()
    
    return

def plot_outages_on_map_state(start_time,
                              end_time,
                              data_directory = './data/eaglei_data',
                              state='Texas'):
    """
    Plots power outages on a map for a specified state and time range.

    Parameters:
    start_time (str): The start time in ISO format (YYYY-MM-DDTHH:MM:SS).
    end_time (str): The end time in ISO format (YYYY-MM-DDTHH:MM:SS).
    data_directory (str, optional): The directory containing the CSV files. Default is './data/eaglei_data'.
    state (str, optional): The state for which to plot the outages. Default is 'Texas'.

    Raises:
    ValueError: If the year extracted from start_time is not between 2014 and 2023.
    FileNotFoundError: If the CSV file for the specified year does not exist in the data_directory.

    Returns:
    None
    """
    
    # Convert start_time and end_time to a datetime object and extract the year
    start_time = datetime.fromisoformat(start_time)
    end_time = datetime.fromisoformat(end_time)
    year = start_time.year
    
    # Check if the year is within the valid range
    valid_years = [str(y) for y in range(2014, 2024)]
    if str(year) not in valid_years:
        raise ValueError(f"Invalid year: {year}. Year must be between 2014 and 2023.")
    
    # Construct the file path
    file_name = f"eaglei_outages_{year}.csv"
    file_path = os.path.join(data_directory, file_name)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_name} does not exist in the directory {data_directory}.")
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    df['run_start_time'] = pd.to_datetime(df['run_start_time'])
    df.dropna(subset=['customers_out'],inplace=True)
    
    
    url = "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_20m.zip"
    counties = gpd.read_file(url)

    counties = counties[counties['STATE_NAME'] == state]
    

    # Filter data based on time range and state
    filtered_df = df[(df['run_start_time'] >= start_time) & (df['run_start_time'] <= end_time) & (df['state'] == state)]
    
    grouped = filtered_df.groupby(['county', 'fips_code']).agg({'customers_out': 'sum'}).reset_index()
    grouped['customers_out'] = grouped['customers_out'] / 96
    
    
    # Merge the Texas counties with aggregated data
    grouped['fips_code'] = grouped['fips_code'].astype(str)  # Ensure FIPS codes are strings
    counties = counties.merge(grouped, left_on='NAME', right_on='county', how='left')
    
    # Fill missing values with zero so we display it in white color
    counties['customers_out'] = counties['customers_out'].fillna(0) 


    # Create a custom colormap
    viridis_cmap = plt.cm.viridis  # Base colormap
    colors = viridis_cmap(np.linspace(0, 1, 256))  # Extract colors from viridis
    colors[0] = np.array([1, 1, 1, 1])  # Set the first color (for zero values) to white
    custom_cmap = ListedColormap(colors)
    
    # Plot the map
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    county_plot = counties.plot(column='customers_out',  # Data to visualize
                                cmap=custom_cmap,  # Colormap
                                linewidth=0.8,
                                ax=ax,
                                edgecolor='black',
                                legend=False,  # Disable automatic colorbar
                                norm=Normalize(vmin=0, vmax=counties['customers_out'].max())  # Normalize scale
                               )
    
    # Add colorbar and set its title
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=Normalize(vmin=0, vmax=counties['customers_out'].max()))
    sm._A = []  # Dummy array for the ScalarMappable
    cbar = fig.colorbar(sm, ax=ax)
    # cbar.set_label('Number of Customers Out', fontsize=12, labelpad=10)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.set_xlabel('customer.day', fontsize=14)
    
    # Remove the vertical label
    cbar.ax.set_ylabel('')
    
    # Customize the plot
    title = f"Power outage in {state}\nfrom {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}"
    ax.set_title(title, fontsize=14)
    ax.axis('off')  # Turn off the axis
    plt.show()

    return

def plot_outages_ts_states(start_time,
                           end_time,
                           states,
                           data_directory = './data/eaglei_data'):
    """
    Plots the time series of total customers without power for multiple states within a specified time range.

    Parameters:
    start_time (str): The start time in ISO format (YYYY-MM-DDTHH:MM:SS).
    end_time (str): The end time in ISO format (YYYY-MM-DDTHH:MM:SS).
    states (list of str): A list of state names to plot.
    data_directory (str, optional): The directory where the data files are stored. Default is './data/eaglei_data'.

    Raises:
    ValueError: If the year extracted from start_time is not between 2014 and 2023.
    FileNotFoundError: If the data file for the specified year does not exist in the data_directory.

    Returns:
    None
    """
    
    # Convert start_time and end_time to a datetime object and extract the year
    start_time = datetime.fromisoformat(start_time)
    end_time = datetime.fromisoformat(end_time)
    year = start_time.year
    
    # Check if the year is within the valid range
    valid_years = [str(y) for y in range(2014, 2024)]
    if str(year) not in valid_years:
        raise ValueError(f"Invalid year: {year}. Year must be between 2014 and 2023.")
    
    # Construct the file path
    file_name = f"eaglei_outages_{year}.csv"
    file_path = os.path.join(data_directory, file_name)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_name} does not exist in the directory {data_directory}.")
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    df['run_start_time'] = pd.to_datetime(df['run_start_time'])
    df.dropna(subset=['customers_out'],inplace=True)

    plt.figure(figsize=(8, 4))

    # Loop through each state and plot the data
    for state in states:
        df_state = df[df['state'] == state]
        
        # Group by 'run_start_time' and sum the 'customers_out' for each timestamp
        df_state_grouped = df_state.groupby('run_start_time')['customers_out'].sum().reset_index()
        
        # Plot the time history of total customers out
        plt.plot(df_state_grouped['run_start_time'], df_state_grouped['customers_out'], linestyle='-', label=state)
    
    plt.title('Total Number of Customers Without Power', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Total Customers Out', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)  # Add legend with specified font size
    plt.show()

    return

def plot_outages_ts_years(state, years, data_directory='./data/eaglei_data'):
    """
    Plots the time series of total customers without power for a specific state over multiple years.

    Parameters:
    state (str): The name of the state to plot.
    years (list of str): A list of years to plot. Each year must be between 2014 and 2023.
    data_directory (str, optional): The directory where the data files are stored. Default is './data/eaglei_data'.

    Raises:
    ValueError: If any year in the list is not between 2014 and 2023.
    FileNotFoundError: If the data file for any specified year does not exist in the data_directory.

    Returns:
    None
    """
    
    # Check if all years are within the valid range
    valid_years = [str(y) for y in range(2014, 2024)]
    for year in years:
        if year not in valid_years:
            raise ValueError(f"Invalid year: {year}. Year must be between 2014 and 2023.")
    
    plt.figure(figsize=(8, 4))

    # Loop through each year and plot the data
    for year in years:
        # Construct the file path
        file_name = f"eaglei_outages_{year}.csv"
        file_path = os.path.join(data_directory, file_name)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_name} does not exist in the directory {data_directory}.")
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        df['run_start_time'] = pd.to_datetime(df['run_start_time'])
        df.dropna(subset=['customers_out'], inplace=True)
        
        # Filter data for the specified state
        df_state = df[df['state'] == state].copy()
        
        # Remove the year from 'run_start_time' to stack plots on top of each other
        df_state['run_start_time'] = df_state['run_start_time'].apply(lambda x: x.replace(year=2000))
        
        # Group by 'run_start_time' and sum the 'customers_out' for each timestamp
        df_state_grouped = df_state.groupby('run_start_time')['customers_out'].sum().reset_index()
        
        # Plot the time history of total customers out
        plt.plot(df_state_grouped['run_start_time'], df_state_grouped['customers_out'], linestyle='-', label=year)
    
    plt.title(f'Total Number of Customers Without Power in {state}', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Total Customers Out', fontsize=14)
    plt.grid(True)
    # Format x-axis to show only month and day, remove year
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
    plt.legend(fontsize=12)  # Add legend with specified font size
    plt.show()

    return

def barchart_events_by_state(df, state, event_types, all_years=True, which_year=None):
    """
    Generates bar charts of event counts by month for a given state and event types.

    Parameters:
    df (pd.DataFrame): DataFrame containing the event data.
    state (str): The state for which the bar charts are to be generated.
    event_types (list): List of event types to be plotted.
    all_years (bool, optional): If True, includes all years in the data. Defaults to True.
    which_year (int, optional): Specific year to filter the data if all_years is False. Defaults to None.

    Returns:
    None: Displays the bar charts.
    """
        
    if all_years:
        df_state = df[df['STATE']==state]
        title = f"{state} 2014-2024"
    else:
        df_state = df[(df['STATE']==state) & (df['YEAR']==which_year)]
        title = f"{state} {which_year}" 
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Loop through event types and create plots
    for i, event in enumerate(event_types):
        # Filter data for the specific event type
        event_data = df_state[df_state['EVENT_TYPE'] == event]        
        
        # Group by MONTH and count events
        monthly_data = event_data.groupby('MONTH_NAME').size().reset_index(name='Event Count')
   
        monthly_data['MONTH_SORT'] = pd.to_datetime(monthly_data['MONTH_NAME'], format='%B')
        monthly_data = monthly_data.sort_values(by='MONTH_SORT')
    
        # Plot the data
        axes[i].bar(monthly_data['MONTH_NAME'], monthly_data['Event Count'], color='skyblue')
        axes[i].set_title(f'{event} Distribution', fontsize=20)
        axes[i].set_xlabel('Month', fontsize=20)
        axes[i].set_ylabel('Number of Events', fontsize=20)
        
        axes[i].set_xticks(range(len(monthly_data['MONTH_NAME'].unique())))
        axes[i].set_xticklabels(monthly_data['MONTH_NAME'].unique(), rotation=45, ha='right', fontsize=15)

    # Set common Y label
    axes[0].set_ylabel('Number of Events', fontsize=15)

    plt.suptitle(title, fontsize=25)
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    return

def make_ts_events(state, event_types, start_year, start_month, start_day, end_year, end_month, end_day, df):
    """
    Construct a DataFrame with 15-minute intervals indicating event occurrence.
   
    Parameters:
    - df (pd.DataFrame): The NOAA StormEvent database.
    - state (str): The state to filter (e.g., "Texas").
    - event_types (list): The event types to filter (e.g., ["Winter Storm", "Hurricane"]).
    - start_year (int): The start year for the new DataFrame.
    - start_month (int): The start month for the new DataFrame.
    - start_day (int): The start day for the new DataFrame.
    - end_year (int): The end year for the new DataFrame.
    - end_month (int): The end month for the new DataFrame.
    - end_day (int): The end day for the new DataFrame.
   
    Returns:
    - pd.DataFrame: A DataFrame with 15-minute intervals and event counts.
    """
    # Generate the time range for the new DataFrame
    start_date = datetime(start_year, start_month, start_day)
    end_date = datetime(end_year, end_month, end_day, 23, 45)  # Include the last time interval
    time_index = pd.date_range(start=start_date, end=end_date, freq='15min')
   
    # Create the new DataFrame with 15-minute intervals
    new_df = pd.DataFrame({'time': time_index})
    
    # Initialize event count columns for each event type
    for event_type in event_types:
        new_df[f'event_count {event_type}'] = 0  # Initialize event counts to 0
   
    # Convert BEGIN and END times into datetime objects
    df['BEGIN_DATETIME'] = pd.to_datetime(
        df['BEGIN_YEARMONTH'].astype(str) + df['BEGIN_DAY'].astype(str).str.zfill(2) +
        df['BEGIN_TIME'].astype(str).str.zfill(4), format='%Y%m%d%H%M'
    )
    df['END_DATETIME'] = pd.to_datetime(
        df['END_YEARMONTH'].astype(str) + df['END_DAY'].astype(str).str.zfill(2) +
        df['END_TIME'].astype(str).str.zfill(4), format='%Y%m%d%H%M'
    )
    
    # Filter the NOAA data for the specified state and event type
    filtered_df = df[
        (df['STATE'] == state) & 
        (df['EVENT_TYPE'].isin(event_types)) & 
        (df['END_DATETIME'] >= start_date) & 
        (df['BEGIN_DATETIME'] <= end_date)
    ].copy(deep=True)
   
    # Iterate through the events and assign them to the closest time interval in the new DataFrame
    for event_type in event_types:
        event_subset = filtered_df[filtered_df['EVENT_TYPE']==event_type]
        
        for _, row in event_subset.iterrows():
            event_start = row['BEGIN_DATETIME']
            event_end = row['END_DATETIME']
       
            # Round the start and end times to the nearest 15-minute interval
            event_start_rounded = event_start.round('15min')
            event_end_rounded = event_end.round('15min')
       
            # Find the indices in the new DataFrame for the rounded times
            start_idx = new_df['time'].searchsorted(event_start_rounded)
            end_idx = new_df['time'].searchsorted(event_end_rounded)
       
            # Increment the event count for the affected time intervals
            if start_idx < len(new_df) and end_idx <= len(new_df):
                new_df.loc[start_idx:end_idx, f'event_count {event_type}'] += 1
   
    # Add YEAR, MONTH, DAY columns
    new_df['YEAR'] = new_df['time'].dt.year
    new_df['MONTH'] = new_df['time'].dt.month
    new_df['DAY'] = new_df['time'].dt.day
    
    # Re-order the columns to make sure the YEAR, MONTH, DAY, time start first
    cols_order = ['YEAR', 'MONTH', 'DAY', 'time'] + [col for col in new_df.columns if col not in ['YEAR', 'MONTH', 'DAY', 'time']]
    new_df = new_df[cols_order]

    
    # Return the new_df
    return new_df

def make_ts_power(state,
                  start_year,
                  start_month,
                  start_day,
                  end_year,
                  end_month,
                  end_day,
                  data_directory = './data/eaglei_data'):
    """
    Generate a time series dataframe of power outages for a specific state within a given date range.

    This function reads yearly CSV files containing power outage data, filters the data for a specified state,
    aggregates the number of customers without power, and returns a time series dataframe for the specified date range.

    Parameters:
    state (str): The state for which to generate the time series data.
    start_year (int): The starting year of the date range.
    start_month (int): The starting month of the date range.
    start_day (int): The starting day of the date range.
    end_year (int): The ending year of the date range.
    end_month (int): The ending month of the date range.
    end_day (int): The ending day of the date range.
    data_directory (str, optional): The directory containing the CSV files. Default is './data/eaglei_data'.

    Returns:
    pd.DataFrame: A dataframe with a datetime index ('time') and a column 'customers_out' representing the number of customers without power.

    Raises:
    FileNotFoundError: If any of the CSV files for the specified years do not exist in the data directory.
    """
    
    df_list = []
    for year in range(start_year, end_year + 1):
        # Construct the filename
        file_name = f"eaglei_outages_{year}.csv"
        file_path = os.path.join(data_directory, file_name)
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_name} does not exist in the directory {data_directory}.")
    
        # Read the CSV file into a dataframe
        df = pd.read_csv(file_path)
        df['run_start_time'] = pd.to_datetime(df['run_start_time'])
        df.dropna(subset=['customers_out'],inplace=True)
        
        df_state = df[df['state']==state].copy(deep=True)
        df_state_ts_cus = df_state.groupby('run_start_time')['customers_out'].sum().reset_index()
        df_state_ts_cus.drop(df_state_ts_cus.index[-1], inplace=True)
        df_state_ts_cus.set_index('run_start_time', inplace=True)
        df_state_ts_cus.rename_axis('time', inplace=True)
    
        
        # Append the dataframe to the list
        df_list.append(df_state_ts_cus)
    
    # Concatenate all dataframes in the list into a single dataframe
    concat_df = pd.concat(df_list, ignore_index=False)
    start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day+1)
    # Slice the dataframe
    df_state_ts_power = concat_df.loc[start_date:end_date].copy(deep=True)
    df_state_ts_power.drop(df_state_ts_power.index[-1], inplace=True)

    return df_state_ts_power

def aggregate_ts(df, agg_type):
    """
    Aggregates the time series data in the dataframe based on the specified aggregation type.

    Parameters:
    df (pd.DataFrame): The input dataframe with datetime index and one or more columns.
    agg_type (str): The aggregation type, either 'hour' or 'day'.

    Returns:
    pd.DataFrame: The aggregated dataframe.
    """
    # Ensure the index is in datetime format
    df.index = pd.to_datetime(df.index)
    
    if agg_type == 'hour':
        # Group by hour and calculate the mean for each hour
        df_agg = df.groupby(pd.Grouper(freq='h')).mean()
    elif agg_type == 'day':
        # Group by day and calculate the mean for each day
        df_agg = df.groupby(pd.Grouper(freq='D')).mean()
    else:
        raise ValueError("Invalid aggregation type. Use 'hour' or 'day'.")
    
    df_agg.fillna(0, inplace=True)
    
    return df_agg

def combine_agg_ts(state,
                   start_year,
                   start_month,
                   start_day,
                   end_year,
                   end_month,
                   end_day,
                   data_directory_power = './data/eaglei_data',
                   data_directory_events = './data/NOAA_StormEvents'):
    """
    Combine and aggregate time series data of power outages and weather events for a specific state within a given date range.

    This function generates time series data for power outages and weather events, aggregates them by hour and day,
    and merges the aggregated data into two combined dataframes.

    Parameters:
    state (str): The state for which to generate the time series data.
    start_year (int): The starting year of the date range.
    start_month (int): The starting month of the date range.
    start_day (int): The starting day of the date range.
    end_year (int): The ending year of the date range.
    end_month (int): The ending month of the date range.
    end_day (int): The ending day of the date range.
    data_directory_power (str, optional): The directory containing the power outage CSV files. Default is './eaglei_data'.
    data_directory_events (str, optional): The directory containing the weather events CSV files. Default is './NOAA_StormEvents'.

    Returns:
    tuple: Two dataframes - the first aggregated by hour and the second aggregated by day, both containing combined time series data of power outages and weather events.

    Raises:
    FileNotFoundError: If any of the required CSV files do not exist in the specified directories.
    """

    df_state_ts_power = make_ts_power(state = state,
                                      start_year = start_year,
                                      start_month = start_month,
                                      start_day = start_day,
                                      end_year = end_year,
                                      end_month = end_month,
                                      end_day = end_day,
                                      data_directory = data_directory_power)
    
    df_state_ts_power_hr = aggregate_ts(df_state_ts_power, 'hour')
    df_state_ts_power_day = aggregate_ts(df_state_ts_power, 'day')
    
    
    
    df_events = pd.read_csv(os.path.join(data_directory_events, "StormEvents_2014_2024.csv"))
    df_state_events=df_events[df_events['STATE']==state.upper()].copy(deep=True)
    event_types_state = list(df_state_events['EVENT_TYPE'].unique())
    
    df_state_ts_events = make_ts_events(state = state.upper(),
                                        event_types= event_types_state,
                                        start_year = start_year,
                                        start_month = start_month,
                                        start_day = start_day,
                                        end_year = end_year,
                                        end_month = end_month,
                                        end_day = end_day,
                                        df=df_events)
    df_state_ts_events['time'] = pd.to_datetime(df_state_ts_events['time'])
    df_state_ts_events.set_index('time', inplace=True)
    df_state_ts_events.drop(columns=['YEAR', 'DAY', 'MONTH'], inplace=True)

    df_state_ts_events_hr = aggregate_ts(df_state_ts_events, 'hour')
    df_state_ts_events_day = aggregate_ts(df_state_ts_events, 'day')

    df_state_ts_comb_hr = pd.merge(df_state_ts_events_hr, df_state_ts_power_hr, left_index=True, right_index=True)
    df_state_ts_comb_day = pd.merge(df_state_ts_events_day, df_state_ts_power_day, left_index=True, right_index=True)
    
    return df_state_ts_comb_hr, df_state_ts_comb_day

def plot_ts_events_power(df, event_types, start_year, start_month, start_day, end_year, end_month, end_day):
    """
    Plots time histories of specified event types and power outages.

    Parameters:
    df (pd.DataFrame): DataFrame with a datetime index.
    event_types (list): List of column names for event counts, e.g., ['event_count Winter Storm', 'event_count Cold/Wind Chill'].
    start_year (int): Start year for the time range.
    start_month (int): Start month for the time range.
    start_day (int): Start day for the time range.
    end_year (int): End year for the time range.
    end_month (int): End month for the time range.
    end_day (int): End day for the time range.

    Returns:
    None
    """
    # Define the time range
    start_date = f"{start_year}-{start_month:02d}-{start_day:02d}"
    end_date = f"{end_year}-{end_month:02d}-{end_day:02d}"
    
    # Filter the dataframe for the specified time range
    df_filtered = df.loc[start_date:end_date]
    
    # Create a plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot event types on the left y-axis
    label_fontsize=14
    tick_fontsize=14
    colors = plt.cm.tab10.colors  # Get a list of colors
    for i, event in enumerate(event_types):
        ax1.plot(df_filtered.index, df_filtered[event], label=event.replace('event_count ', ''), 
                 color=colors[i % len(colors)], linewidth=2.0, alpha=0.7, linestyle='-.')  # Thicker and semi-transparent lines with dashed-dot style
    ax1.set_xlabel('Date', fontsize=label_fontsize)
    ax1.set_ylabel('Event Counts', fontsize=label_fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(loc='upper left', fontsize = 12)
    
    # Create a second y-axis for customers_out
    ax2 = ax1.twinx()
    ax2.plot(df_filtered.index, df_filtered['customers_out'], color='red', linewidth=2, linestyle='-')  # Thicker line with dashed-dot style
    ax2.set_ylabel('Customers Out', color='red', fontsize=label_fontsize)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=tick_fontsize)
    
    # Set the title
    plt.title('Time Histories of Event Counts and Power Outages', fontsize=label_fontsize)
    
    # Show the plot
    plt.show()
    
    return
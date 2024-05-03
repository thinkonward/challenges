import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import random
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape
from datetime import datetime


def calculate_average_hourly_energy_consumption(folder_path, season_months_dict):
    """
    Process multiple parquet files in a folder, calculate hourly average energy consumption,
    and return a pandas DataFrame with each row corresponding to one file in the folder.

    Parameters:
    - folder_path (str): Path to the folder containing parquet files.
    - season_months_dict (dict): A dictionary where keys are season names (strings) and values are lists
      of corresponding month numbers. For example, {'cold': [1, 2, 12], 'hot': [6, 7, 8], 'mild': [3, 4, 5, 9, 10, 11]}.

    Returns:
    - df_ave_hourly (pd.DataFrame): A pandas DataFrame with each row corresponding to one file in the folder (i.e. one building).
      The columns are multi-layer with the first layer being the season and the second layer the hour of the day 
      Index ('bldg_id') contains building IDs. Column values are average hourly electricity energy consumption
    """
    # Initialize an empty list to store individual DataFrames for each file
    result_dfs = []

    # Iterate through all files in the folder_path
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".parquet"):
            # Extract the bldg_id from the file name
            bldg_id = int(file_name.split('.')[0])

            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)

            # Read the original parquet file
            df = pd.read_parquet(file_path)

            # Convert 'timestamp' column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Extract month and hour from 'timestamp'
            df['month'] = df['timestamp'].dt.month
            df['hour'] = df['timestamp'].dt.hour

             # Create a mapping from month to the corresponding season
            month_to_season = {month: season for season, months_list in season_months_dict.items() for month in months_list}

            # Assign a season to each row based on the month
            df['season'] = df['month'].map(month_to_season)

            # Calculate hourly average energy consumption for each row
            df['hourly_avg_energy_consumption'] = 4 * df.groupby(['season', 'hour'])['out.electricity.total.energy_consumption'].transform('mean')

            # Pivot the dataframe to create the desired output format
            result_df = df.pivot_table(values='hourly_avg_energy_consumption', index='bldg_id', columns=['season', 'hour'])

            # Reset the column names
            result_df.columns = pd.MultiIndex.from_tuples([(season, hour+1) for season, months_list in season_months_dict.items() for hour in range(24)])

            # Add 'bldg_id' index with values corresponding to the names of the parquet files
            result_df['bldg_id'] = bldg_id
            result_df.set_index('bldg_id', inplace=True)

            # Append the result_df to the list
            result_dfs.append(result_df)

    # Concatenate all individual DataFrames into a single DataFrame
    df_ave_hourly = pd.concat(result_dfs, ignore_index=False)

    return df_ave_hourly

def add_building_tags(df):
    """
    Add building tags from an auxiliary CSV file to the input DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing the original data in the auxiliary_data folder.

    Returns:
    - df_with_tags (pandas.DataFrame): A new DataFrame with an additional column
      ('building_tags', '') containing building tags for each building.
    """

    csv_file_path = 'auxiliary_data_building_tags.csv'

    # Reading the CSV file containing the building tags
    df_tags = pd.read_csv(csv_file_path, index_col=0,  header=0, skiprows=[0, 1], names = ['building_tags'])
    
    # Convert the string representations of dictionaries to actual dictionaries
    df_tags['building_tags'] = df_tags['building_tags'].apply(literal_eval)

    multi_index = pd.MultiIndex.from_tuples([('building_tags', '')])

    # Assign the MultiIndex to the DataFrame columns
    df_tags.columns = multi_index

    df_with_tags = pd.concat([df,df_tags], axis=1)
    
    return df_with_tags

def get_building_tag_values(df, tag_key):
    """
    Extracts building_ids for each unique value of the specified tag key in the 'building_tags' column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with multi-level columns and 'bldg_id' index.
    - tag_key (str): The key to extract values from the dictionaries in the 'building_tags' column.

    Returns:
    dict: A dictionary where keys are unique values of the specified tag key,
          and values are lists of building_ids associated with each unique tag value.
    """

    tag_values_dict = {}

    for index, row in df.iterrows():
        # Check if the 'building_tags' column has a dictionary
        if isinstance(row[('building_tags', '')], dict):
            tag_value = row[('building_tags', '')].get(tag_key)
            
            # Add the building_id to the corresponding tag value in the dictionary
            if tag_value is not None:
                tag_values_dict.setdefault(tag_value, []).append(index)

    return tag_values_dict

def normalize_df(df, normalize = None):
    """
    Normalize a DataFrame based on the specified normalization method.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame to be normalized.
    - normalize (str or None, optional): Normalization method.
      Choose from [None, 'within_season', 'within_year'].
      When set to 'within_season' or 'within_year', values in the df are normalized
      (divided by the max value) within the season or year, respectively, for each row.
      Default is None.

    Returns:
    - df_normalized (pandas.DataFrame): A normalized DataFrame based on the chosen method.
    """
    
    if normalize not in [None, 'within_season', 'within_year']:
        raise ValueError("Invalid value for normalize. Choose from [None, 'within_season', 'within_year']")
    
    df1 = df.copy()
    # Normalize the dataframe
    if normalize == 'within_season':
        df_normalized = df1.div(df.groupby(axis=1, level=0).transform('max'), axis=0)
    elif normalize == 'within_year':
        df_normalized = df1.div(df.max(axis=1), axis=0)
    else:
        df_normalized = df1.copy()
    
    return df_normalized

def fit_kshape_clustering(n_clusters, X, num_seeds=10, verbose = False):
    """
    Perform kShape clustering with different random seeds and return the best model.

    Parameters:
    - n_clusters (int): Number of clusters to form.
    - X (numpy.ndarray): Input time series data of shape (n_samples, n_features).
    - num_seeds (int, optional): Number of random seeds to try. Default is 10.
    verbosebool (default: False): Whether or not to print information about the inertia while learning the model.

    Returns:
    - best_model (tslearn.clustering.kshape.KShape): Best kShape clustering model based on evaluation metric of inertia.
    """
    best_model = None
    best_score = float('inf')

    for seed in range(num_seeds):
        np.random.seed(seed)
        
        # Preprocess the input data
        X_scaled = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X)
        sz = X_scaled.shape[1]

        # kShape clustering
        ks = KShape(n_clusters=n_clusters, verbose=verbose, random_state=seed)
        ks.fit(X_scaled)

        # Evaluate the clustering model (replace with your own metric if needed)
        score = ks.inertia_

        # Check if the current model is better than the best
        if score < best_score:
            best_score = score
            best_model = ks

    return best_model, X_scaled

def plot_kshape_clustering(X, ks, df_with_tags, bldg_ids_list, tag_key):

    """
    Plot kShape clustering results with additional information about building tags.

    Parameters:
    - X (numpy.ndarray): Input time series data of shape (n_samples, n_features, 1).
    - ks (tslearn.clustering.kshape.KShape): Fitted kShape clustering model.
    - df_with_tags (pandas.DataFrame): DataFrame containing building tags for each building.
    - bldg_ids_list (list): List of building IDs corresponding to the input time series data.
    - tag_key (str): Key to access the building tag information in df_with_tags.

    Returns:
    None (displays plots).

    The function plots the clustering results for each cluster along with additional information
    about building tags (the count of buildings with each tag in the cluster).

    It uses kShape clustering results to plot individual time series and cluster centers.

    Additionally, a text box is added next to each plot, providing information about the building tags
    in the corresponding cluster, including tag names and the count of buildings with each tag.

    """

    y_pred = ks.predict(X)
    n_clusters = ks.cluster_centers_.shape[0]
    sz = X.shape[1]
    
    fig, axs = plt.subplots(n_clusters, 1)

    for yi, ax in zip(range(n_clusters), axs):
        for xx in X[y_pred == yi]:
            ax.plot(xx.ravel(), "k-", alpha=.2)
        ax.plot(ks.cluster_centers_[yi].ravel(), "r-")
        ax.set_xlim(0, sz)
        ax.set_title("Cluster %d" % (yi + 1))
    
        # Create a text box with information about keys and lengths
        tag_values_dict_cluster = get_building_tag_values(df = df_with_tags.loc[bldg_ids_list][y_pred == yi], tag_key = tag_key)
        text = '\n'.join([f'{key}: {len(value)}' for key, value in tag_values_dict_cluster.items()])
        ax.text(1.05, 0.5, text, transform=ax.transAxes, va='center', ha='left', fontsize=12, color='black')
    
    plt.tight_layout(rect=[0, 0, 1, 1.05])  # Adjust layout to make room for the text box
    plt.show()




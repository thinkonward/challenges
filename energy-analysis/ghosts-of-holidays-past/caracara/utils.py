import pandas as pd
import random
from datetime import datetime
import numpy as np

def scoring(submission_file, groundtruth_partycost_file):
    """
    Calculates the error score for the submission file.

    Args:
    - submission_file: Filepath including filename and .csv extension for the submission file
    - groundtruth_partycost_file: Filepath including filename and .csv extension for the ground-truth Party cost 
    of the Scrooge House.

    Returns:
    Returns the calculated error score for the submission file.
    """
    
    optimal_cost = 6
    alpha = 1
    beta = 1/16

    df = pd.read_csv(submission_file)
    df_partycost = pd.read_csv(groundtruth_partycost_file)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_partycost['timestamp'] = pd.to_datetime(df_partycost['timestamp'])
    
    first_timestamp = df['timestamp'].iloc[0]
    last_timestamp = df['timestamp'].iloc[-1]
    
    df['groundtruth_cost'] = df_partycost[(df_partycost['timestamp']>=first_timestamp) & (df_partycost['timestamp']<=last_timestamp)]['party_cost'].values

    partycost_rmse = ((df['party_cost'] - df['groundtruth_cost']) ** 2).mean() ** .5
    total_partycost_rmse = abs(df['party_cost'].sum()-optimal_cost)
    
    total_error = alpha * partycost_rmse + beta * total_partycost_rmse
    
    return total_error

def sample_submission_generator(start_datetime, max_party_cost, destination_path):
    """
    Generate a sample submission dataframe and save it as a CSV file.

    Args:
    - start_datetime: Starting datetime in 'YYYY-MM-DD HH:MM:SS' format or as a datetime object.
    - max_party_cost: Maximum value for the entries in 'party_cost' column.
    - destination_path: Filepath including filename and .csv extension to save the generated DataFrame.

    Returns:
    Returns the generated Dataframe and saves it as a CSV file at the specified destination_path.
    """

    # Convert start_datetime to a datetime object if it's a string
    if isinstance(start_datetime, str):
        start_datetime = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')

    # Generate the timestamp values
    timestamps = pd.date_range(start=start_datetime, periods=16, freq='15T')

    # Generate random party_cost values
    party_costs = np.random.uniform(0, max_party_cost, size=16)

    # Create the DataFrame
    df = pd.DataFrame({'timestamp': timestamps, 'party_cost': party_costs})

    # Save the DataFrame to a CSV file
    df.to_csv(destination_path, index=False)
    
    return df
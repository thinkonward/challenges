# utils
import numpy as np
import pandas as pd
from IPython.display import display, HTML


def centered_average(x, half_width=96*15):
    """
    Calculates a centered moving average for the given data.

    Parameters
    ----------
    x : array_like
        Input array or sequence for which the centered moving average is to be calculated.
    half_width : int, optional
        Defines half the window size for the moving average, defaulting to 96*15 which typically represents
        a moving average over a certain number of days given data in hourly increments.

    Returns
    -------
    ndarray
        The centered moving average of the input array, with padding on both ends to match the size of the input array.
    """    
    window_size = 2*half_width+1
    pad_left = [x[:2*i+1].mean() for i in range(half_width)]
    pad_right = [x[-2*i-1:].mean() for i in range(half_width)[::-1]]
    avg = np.convolve(x, np.ones(window_size)/window_size, mode="valid")
    return np.concatenate([pad_left, avg, pad_right])


def load_building_data(building_num):
    """
    Loads and aggregates energy consumption data for a specified building from a parquet file.

    Parameters
    ----------
    building_num : int
        The building number identifier used to locate the corresponding data file.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing hourly aggregated energy consumption data for the specified building, limited to 8760 hours (typically one year).
    """
    df = pd.read_parquet("./data/data/{0}.parquet".format(building_num), engine="pyarrow")
    df = df.rename(columns={"out.electricity.total.energy_consumption": "energy_consumption"})
    df = df.drop(columns="timestamp").groupby(df.timestamp.dt.floor("h")).sum().reset_index()
    df["hour_of_week"] = (df.timestamp.dt.dayofweek * 24 + df.timestamp.dt.hour) % (24 * 7)
    return df.head(8760)


def ecdf(data):
    """
    Computes the empirical cumulative distribution function (ECDF) for a one-dimensional array of measurements.

    Parameters
    ----------
    data : array_like
        An array of measurements for which the ECDF is to be calculated.

    Returns
    -------
    tuple
        A tuple (x, y) where x is the sorted data and y is the ECDF values for each data point in x.
    """
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y


def display_dataframe(df):
    """
    Displays a DataFrame in a Jupyter notebook without row indices.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to display.

    Returns
    -------
    None
    """
    display(HTML(df.to_html(index=False)))


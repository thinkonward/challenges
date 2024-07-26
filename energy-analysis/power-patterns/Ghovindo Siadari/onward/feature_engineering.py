# feature engineering
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from onward.utils import centered_average


def get_size_feature(data_dict):
    """
    Computes the total energy consumption for each building in a dataset.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing building data, where each key corresponds to a building identifier and its value is a DataFrame of energy consumption data.

    Returns
    -------
    ndarray
        An array containing the total energy consumption for each building in the dataset.
    """    
    return np.array([data_dict[i].energy_consumption.sum() for i in range(1, 1278)])


def get_seasonal_features(data_dict):
    """
    Calculates normalized seasonal average profiles of energy consumption for each building.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing building data, each key corresponds to a building identifier and its value is a DataFrame of energy consumption data.

    Returns
    -------
    raw_seasonal_features : ndarray
        An array of raw seasonal energy consumption profiles for each building.
    seasonal_features : ndarray
        An array of normalized seasonal energy consumption profiles for each building.
    """
    raw_seasonal_features = []
    seasonal_features = []
    for building_num in range(1, 1278):

        # calculate seasonal average
        df = data_dict[building_num]
        seasonal_profile = df.groupby(df.timestamp.dt.date)["energy_consumption"].mean().reset_index()
        seasonal_profile = centered_average(seasonal_profile.energy_consumption, half_width=15)
        raw_seasonal_features.append(seasonal_profile)

        # normalize data
        mu, sigma = df.energy_consumption.mean(), df.energy_consumption.std()
        seasonal_profile = (seasonal_profile - mu) / sigma
        seasonal_features.append(seasonal_profile)

    return np.array(raw_seasonal_features), np.array(seasonal_features)


def get_daily_features(data_dict):
    """
    Computes normalized daily average energy consumption profiles for each building.

    Parameters
    ----------
    data_dict : dict
        Dictionary where each key corresponds to a building identifier and its value is a DataFrame of energy consumption data.

    Returns
    -------
    raw_daily_features : ndarray
        An array containing raw daily energy consumption profiles for each building.
    daily_features : ndarray
        An array containing normalized daily energy consumption profiles for each building.
    """
    raw_daily_features = []
    daily_features = []
    for building_num in range(1, 1278):

        # calculate daily average
        df = data_dict[building_num]
        daily_profile = df.groupby(df.hour_of_week % 24)["energy_consumption"].mean().values
        raw_daily_features.append(daily_profile)

        # normalize data
        mu, sigma = df.energy_consumption.mean(), df.energy_consumption.std()
        daily_profile = (daily_profile - mu) / sigma
        daily_features.append(daily_profile)

    return np.array(raw_daily_features), np.array(daily_features)


def get_final_cluster_info(cluster_data):
    """
    Calculates aggregate building statistics for each cluster.

    Parameters
    ----------
    cluster_data : DataFrame
        DataFrame containing the cluster groups and total energy consumption for each building.

    Returns
    -------
    DataFrame
        A DataFrame containing aggregate statistics for each cluster, including total consumption, average consumption,
        count, and building identifiers.
    """

    cluster_info = []

    # large winter/summer midday peak
    for group in [
        ("large", "midday_peak", "summer_peak"),
        ("large", "midday_peak", "winter_peak"),
    ]:

        df = cluster_data.loc[
            (cluster_data.total_consumption_group == group[0]) &
            (cluster_data.time_of_day_consumption_group == group[1])]

        if "summer" in group[2]:
            name = "{0} Summer {1}".format(group[0], group[1])
            df = df.loc[df.summer_consumption_group == group[2]]
        else:
            name = "{0} Winter {1}".format(group[0], group[1])
            df = df.loc[df.winter_consumption_group == group[2]]

        total = df.total_energy_consumption.sum()
        cluster_info.append(dict(
            name=name.replace("_", " ").title(),
            size=total / 1e6, avg=total / 1e6 / len(df), count=len(df),
            ids=df.building_num.tolist(),
        ))

    # large midday trough
    df = cluster_data.loc[cluster_data.total_consumption_group == "large"]
    df = df.loc[df.time_of_day_consumption_group == "midday_trough"]
    total = df.total_energy_consumption.sum()
    cluster_info.append(dict(
        name="Large Midday Trough",
        size=total / 1e6, avg=total / 1e6 / len(df), count=len(df),
        ids=df.building_num.tolist(),
    ))

    # evening peak
    for group in ["summer_peak", "winter_peak"]:

        df = cluster_data.loc[cluster_data.time_of_day_consumption_group == "evening_peak"]

        if group == "summer_peak":
            name = "Summer Evening Peak"
            df = df.loc[cluster_data.summer_consumption_group == "summer_peak"]
        else:
            name = "Winter Evening Peak"
            df = df.loc[cluster_data.winter_consumption_group == "winter_peak"]

        total = df.total_energy_consumption.sum()
        cluster_info.append(dict(
            name=name,
            size=total / 1e6, avg=total / 1e6 / len(df), count=len(df),
            ids=df.building_num.tolist(),
        ))



    return pd.DataFrame(cluster_info).sort_values("avg", ignore_index=True, ascending=False)

from datetime import date
from typing import Optional

import holidays
import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from mlforecast.feature_engineering import transform_exog
from numba import njit
from sklearn.preprocessing import LabelEncoder
from window_ops.rolling import rolling_mean
from window_ops.shift import shift_array
from window_ops.utils import first_not_na, _validate_rolling_sizes

from src.settings import LAT_COL, LON_COL, BUILDING_ID_COL, HEATING_COL, HEATING_BKUP_COL, HEATING_TOTAL_COL, PLUG_COL, \
    HEATING_AND_PLUG_COL, ALL_RAW_TARGETS

SUN_FIELDS = ["dawn", "sunrise", "noon", "sunset", "dusk"]

weather_columns = ["temperature_2m",
                   "relative_humidity_2m",
                   "wind_speed_10m",
                   "wind_direction_10m",
                   "direct_radiation_instant",
                   "diffuse_radiation_instant",
                   "direct_normal_irradiance_instant",
                   "global_tilted_irradiance_instant",
                   "temperature_setpoint_diff",
                   "temperature_setpoint_ratio"
                   ]


def add_holiday_features(df_sorted):
    ny_holidays = holidays.country_holidays('US', subdiv='NY')
    df_sorted["holiday"] = df_sorted["timestamp"].dt.date.apply(lambda x: x in ny_holidays)

    df_sorted["nearest_holiday"] = df_sorted["timestamp"][df_sorted["holiday"]]
    df_sorted["previous_holiday"] = df_sorted["nearest_holiday"].ffill()
    df_sorted["next_holiday"] = df_sorted["nearest_holiday"].bfill()

    df_sorted["days_from_holiday"] = (df_sorted["timestamp"] - df_sorted["previous_holiday"]).dt.days
    df_sorted["days_to_holiday"] = (df_sorted["next_holiday"] - df_sorted["timestamp"]).dt.days

    df_sorted.drop(columns=["nearest_holiday", "previous_holiday", "next_holiday"], inplace=True)

    return df_sorted


@njit
def ratio_over_previous(x, offset=1):
    """Computes the ratio between the current value and its `offset` lag"""
    return x / shift_array(x, offset=offset)


@njit
def diff_ratio_over_previous(x, offset=1):
    """Computes the ratio between the current value and its `offset` lag"""
    shifted = shift_array(x, offset=offset)
    return x - shifted / x


@njit
def diff_over_previous(x, offset=1):
    """Computes the difference between the current value and its `offset` lag"""
    return x - shift_array(x, offset=offset)


@njit
def rolling_trend(input_array: np.ndarray,
                  window_size: int,
                  min_samples: Optional[int] = None) -> np.ndarray:
    n_samples = input_array.size
    window_size, min_samples = _validate_rolling_sizes(window_size, min_samples)

    output_array = np.full_like(input_array, np.nan)
    start_idx = first_not_na(input_array)
    if start_idx + min_samples > n_samples:
        return output_array

    accum = 0.
    epsilon = 1e-10
    upper_limit = min(start_idx + window_size, n_samples)
    for i in range(start_idx, upper_limit):
        accum += (input_array[i] - input_array[i - 1]) / (input_array[i - 1] + epsilon)
        if i + 1 >= start_idx + min_samples:
            output_array[i] = accum / (i - start_idx + 1)

    for i in range(start_idx + window_size, n_samples):
        value_to_add = (input_array[i] - input_array[i - 1]) / (input_array[i - 1] + epsilon)
        value_to_subtract = (input_array[i - window_size] - input_array[i - 1 - window_size]) / (
                input_array[i - 1 - window_size] + epsilon)
        accum += value_to_add - value_to_subtract
        output_array[i] = accum / window_size

    return output_array


def add_temperature_features(df):
    hour_lags = np.array([2, 3, 4])

    transformed = transform_exog(
        df[[BUILDING_ID_COL, "timestamp"] + weather_columns],
        lags=np.array(range(4, 13)),
        lag_transforms={
            1: [(rolling_mean, 4), (rolling_mean, 8),
                (rolling_trend, 4), (rolling_trend, 8),
                (diff_over_previous, 1), (diff_over_previous, 2),
                (ratio_over_previous, 1), (ratio_over_previous, 2),
                (diff_ratio_over_previous, 1), (diff_ratio_over_previous, 2)
                ],
            2: [(rolling_mean, 4), (rolling_mean, 8),
                (rolling_trend, 4), (rolling_trend, 8),
                (diff_over_previous, 1), (diff_over_previous, 2),
                (ratio_over_previous, 1), (ratio_over_previous, 2),
                (diff_ratio_over_previous, 1), (diff_ratio_over_previous, 2)
                ],
            3: [(rolling_mean, 4), (rolling_mean, 8),
                (rolling_trend, 4), (rolling_trend, 8),
                (diff_over_previous, 1), (diff_over_previous, 2),
                (ratio_over_previous, 1), (ratio_over_previous, 2),
                (diff_ratio_over_previous, 1), (diff_ratio_over_previous, 2)
                ]
        },
        id_col=BUILDING_ID_COL,
        time_col="timestamp"
    )

    return df.merge(transformed.drop(columns=weather_columns),
                    on=[BUILDING_ID_COL, "timestamp"])


def extract_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6])

    df["decimal_hour"] = df["hour"] + df["minute"] / 60

    # # cyclical encoding
    for var in ["dayofweek", "decimal_hour"]:
        df[f"{var}_sin"] = np.sin(df[var] * (2.0 * np.pi / df[var]).max())
        df[f"{var}_cos"] = np.cos(df[var] * (2.0 * np.pi / df[var]).max())

    df = add_temperature_features(df)

    # cat encoding
    df["decimal_hour_cat"] = (df["hour"] + df["minute"] / 60).astype("category")
    df["month_cat"] = (df["timestamp"].dt.month).astype("category")
    df["dayofweek_cat"] = df["dayofweek"].astype("category")
    label_encoder = LabelEncoder()
    df["temperature_cat"] = label_encoder.fit_transform(pd.cut(df["temperature_2m"], 200))

    df = add_holiday_features(df)

    return df


def prepare_dataset(weather_data, electricity_data, metadata):
    bldg_id = "scrooge_bldg"

    electricity_data["timestamp"] = electricity_data["timestamp"].astype("datetime64[ns]")
    electricity_data = electricity_data[["timestamp"] + ALL_RAW_TARGETS].copy()
    weather_data["temperature_setpoint_diff"] = weather_data["temperature_2m"] - 20.55
    weather_data["temperature_setpoint_ratio"] = weather_data["temperature_2m"] / 20.55
    interp_weather = (weather_data
                      .rename(columns={"date": "timestamp"})
                      .set_index("timestamp").resample("15min")
                      .mean().interpolate().reset_index())

    with_weather = electricity_data.merge(interp_weather, on="timestamp", how="right")
    with_weather[BUILDING_ID_COL] = bldg_id
    with_weather[BUILDING_ID_COL] = with_weather[BUILDING_ID_COL].astype('category')

    metadata[BUILDING_ID_COL] = "scrooge_bldg"
    latitude = float(metadata[metadata["bldg_id"] == bldg_id][LAT_COL].values[0])
    longitude = float(metadata[metadata["bldg_id"] == bldg_id][LON_COL].values[0])
    loc = LocationInfo(name='NY', region='NY, USA', timezone='America/New_York',
                       latitude=latitude, longitude=longitude)
    dates = pd.date_range(date(2018, 1, 1), date(2018, 12, 31))
    sun_data = []
    for date_ in dates:
        s = sun(loc.observer, date=date_, tzinfo=loc.timezone)
        sun_data.append(s)
    sun_df = pd.DataFrame(sun_data)
    sun_df["date"] = sun_df["sunrise"].dt.date

    with_weather["date"] = with_weather["timestamp"].dt.date
    with_weather = with_weather.merge(sun_df, on="date").drop(columns="date")

    for field in SUN_FIELDS:
        with_weather[f"time_to_{field}"] = (with_weather[field].dt.tz_localize(None) - with_weather[
            "timestamp"]).dt.total_seconds() / 60

    with_weather.drop(columns=SUN_FIELDS, inplace=True)

    features = extract_features(with_weather)

    features[BUILDING_ID_COL] = features[BUILDING_ID_COL].astype("str")
    features[HEATING_TOTAL_COL] = features[HEATING_COL] + features[HEATING_BKUP_COL]
    features[HEATING_AND_PLUG_COL] = features[HEATING_TOTAL_COL] + features[PLUG_COL]

    print("Finished feature extraction")
    print(features.columns)

    print(features.select_dtypes('object').columns.tolist())

    return features

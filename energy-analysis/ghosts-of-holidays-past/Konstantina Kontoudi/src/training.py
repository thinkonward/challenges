import numpy as np

from src.model import ElectricityModel
from src.settings import ALL_TARGETS, HEATING_TOTAL_COL, PLUG_COL

FEATURES_NAMES = ['heating_total', 'out.electricity.plug_loads.energy_consumption',
                  'temperature_2m_lag12',
                  'wind_speed_10m_rolling_mean_lag1_window_size4', 'wind_speed_10m_rolling_mean_lag1_window_size8',
                  'minute', 'wind_speed_10m_diff_over_previous_lag1', 'temperature_2m', 'dayofweek', 'bldg_id',
                  'days_from_holiday', 'days_to_holiday', 'is_weekend',
                  'holiday', 'temperature_2m_lag8',
                  'hour', 'decimal_hour', 'timestamp',
                  'global_tilted_irradiance_instant_rolling_trend_lag1_window_size8',
                  'global_tilted_irradiance_instant_lag10',
                  'wind_direction_10m_lag10', 'time_to_sunrise', 'wind_direction_10m_rolling_mean_lag1_window_size8',
                  'direct_radiation_instant_rolling_trend_lag1_window_size8',
                  'wind_speed_10m_diff_over_previous_lag1_offset2',
                  'decimal_hour_cat', 'temperature_setpoint_ratio_diff_over_previous_lag1',
                  'relative_humidity_2m_rolling_trend_lag1_window_size4']

RES_MODEL_PARAMS = {
    "n_estimators": 1024,
    "num_leaves": 25,
    "learning_rate": 0.013677456925420968,
    "reg_alpha": 1.2960920082446772,
    "reg_lambda": 0.04043901974593962,
    "colsample_bytree": 0.9835850872376016,
    "subsample": 0.672854630742812,
    "min_child_samples": 37,
    "metric": "rmse",
    "random_state": 42
}


def train(features):
    print("Starting training")

    features = features.dropna(subset=ALL_TARGETS).replace({np.inf: 10_000, -np.inf: 10_000})[FEATURES_NAMES]
    print(f"Number of features: {len(features.columns)}")

    models = {}
    for target_col in [HEATING_TOTAL_COL, PLUG_COL]:
        print(f"Fitting model for variable: {target_col}")
        if target_col == PLUG_COL:
            use_kink = False
        else:
            use_kink = True

        model = ElectricityModel(use_kink=use_kink, temp_column='temperature_2m',
                                 residual_model_params=RES_MODEL_PARAMS)

        X, y = features.drop(columns=[HEATING_TOTAL_COL, PLUG_COL]), features[target_col].values
        model.fit(X, y)
        models[target_col] = model

    return models

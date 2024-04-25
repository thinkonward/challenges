import lightgbm as lgb
import numpy as np

from mlforecast import MLForecast
from mlforecast.target_transforms import Differences

from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class TemperatureModel(BaseEstimator):
    """
    Expects an X with only one column: the temperature
    """

    def __init__(self, use_kink=True, target_transformation=None, temperature_col="temperature_2m"):
        self.linear = LinearRegression()
        self.use_kink = use_kink
        self.best_split_point = None
        self.target_transformation = target_transformation
        self.temperature_col = temperature_col

    def fit(self, X, y, weights=None):

        if self.use_kink:
            temperature_values = X[self.temperature_col].values
            min_temp = temperature_values.min() + 3
            max_temp = temperature_values.max() - 3
            best_score = None
            for t in np.arange(min_temp, max_temp, (max_temp - min_temp) / 20):
                mask = temperature_values < t
                sample_weight = None
                score_weights = None
                if weights is not None:
                    sample_weight = weights[mask]
                    score_weights = np.concatenate([weights[mask], weights[~mask]])
                self.linear.fit(X[[self.temperature_col]][mask], y[mask], sample_weight=sample_weight)
                # compute RMSE for the fit
                score = mean_squared_error(
                    np.concatenate([y[mask], y[~mask]]),
                    np.concatenate([self.linear.predict(X[[self.temperature_col]][mask]),
                                    np.zeros(y[~mask].shape)]),
                    sample_weight=score_weights,
                    squared=False)
                if best_score is None or score < best_score:
                    best_score = score
                    self.best_split_point = t

            mask = temperature_values < self.best_split_point
            self.linear.fit(X[[self.temperature_col]][mask], y[mask])
            # print(self.best_split_point, best_score)
        else:
            self.linear.fit(X[[self.temperature_col]], y)

    def predict(self, X):
        if self.use_kink:
            temperature_values = X[self.temperature_col].values
            mask = temperature_values < self.best_split_point
            predictions = np.zeros(temperature_values.shape)
            if mask.sum() > 0:
                predictions[mask] = self.linear.predict(X[[self.temperature_col]][mask])
        else:
            predictions = self.linear.predict(X[[self.temperature_col]])
        return predictions


class ElectricityModel(BaseEstimator):
    """
    The class expects an X having a column that serves as time series identifier `ts_identifier`
    and a temperature column `temp_column`
    """

    def __init__(self, ts_identifier="bldg_id", temp_column="temperature_2m",
                 use_kink=True, residual_model_params=None):
        self.ts_identifier = ts_identifier
        self.temp_column = temp_column
        self.use_kink = use_kink
        self.temp_models = {}

        self.residuals_ = None
        self.residual_model = lgb.LGBMRegressor()
        self.residual_model_params = residual_model_params

        if residual_model_params is not None:
            self.residual_model.set_params(**residual_model_params)

    def fit(self, X, y):
        temperature_residuals = np.zeros(y.shape)
        for ts_id in X[self.ts_identifier].unique():
            temp_model = TemperatureModel(use_kink=self.use_kink, temperature_col=self.temp_column)
            mask = X[self.ts_identifier] == ts_id
            temp_model.fit(X.loc[mask], y[mask])
            self.temp_models[ts_id] = temp_model
            temperature_residuals[mask] = y[mask] - temp_model.predict(X.loc[mask])

        X = X.drop(columns=["timestamp", self.ts_identifier], errors='ignore')

        self.residuals_ = temperature_residuals
        self.residual_model.fit(X, self.residuals_)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for ts_id in X[self.ts_identifier].unique():
            mask = X[self.ts_identifier] == ts_id
            predictions[mask] = self.temp_models[ts_id].predict(X[mask])

        X = X.drop(columns=["timestamp", self.ts_identifier], errors='ignore')

        return predictions + self.residual_model.predict(X)

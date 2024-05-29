from datetime import datetime
from typing import List, Tuple

import pandas as pd
import requests_cache
import openmeteo_requests
from retry_requests import retry
from rtdip_sdk.pipelines._pipeline_utils.models import Libraries, SystemType
from rtdip_sdk.pipelines.sources.interfaces import SourceInterface

HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoHistoricalWeatherSource(SourceInterface):

    def __init__(
            self,
            latitude: float,
            longitude: float,
            start_date: str,
            end_date: str,
            variables: List[str],
            data_frequency: str,
            timezone: str
    ):
        """
        Downloads data from the open meteo historical weather API.
        To find the list of valid variables refer to the documentation here:
        https://open-meteo.com/en/docs/historical-weather-api
        Args:
            latitude: latitude
            longitude: longitude
            start_date: start date as a "YYYY-mm-dd" formatted string
            end_date: end date as a "YYYY-mm-dd" formatted string
            variables: the list of variable names
            data_frequency: can be hourly or daily
            timezone: the timezone string
        """
        self.latitude = latitude
        self.longitude = longitude
        self.start_date = start_date
        self.end_date = end_date
        self.data_frequency = data_frequency
        self.variables = variables
        self.timezone = timezone
        
    @staticmethod
    def system_type():
        """
        Attributes:
            SystemType (Environment): Requires PYTHON
        """
        return SystemType.PYTHON

    @staticmethod
    def libraries():
        libraries = Libraries()
        return libraries

    @staticmethod
    def settings() -> dict:
        return {}

    def pre_read_validation(self):
        if self.data_frequency not in ["daily", "hourly"]:
            return False
        return True

    def post_read_validation(self):
        return True

    def read_batch(self) -> pd.DataFrame:
        """
        Makes HTTP requests to get data from the Open Meteo API
        """
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        open_meteo = openmeteo_requests.Client(session=retry_session)

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "timezone": self.timezone
        }

        if self.data_frequency == "daily":
            params.update({
                "daily": self.variables,
            })
        else:
            params.update({
                "hourly": self.variables,
            })

        responses = open_meteo.weather_api(HISTORICAL_WEATHER_URL, params=params)

        response = responses[0]

        if self.data_frequency == "hourly":
            # Process hourly data. The order of variables needs to be the same as requested.
            hourly = response.Hourly()
            hourly_data = {"date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )}
            for i, name in enumerate(self.variables):
                hourly_data[name] = hourly.Variables(i).ValuesAsNumpy()

            dataframe = pd.DataFrame(data=hourly_data)
        else:
            # Process daily data. The order of variables needs to be the same as requested.
            daily = response.Daily()
            daily_data = {"date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s"),
                end=pd.to_datetime(daily.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            )}

            for i, name in enumerate(self.variables):
                daily_data[name] = daily.Variables(i).ValuesAsNumpy()

            dataframe = pd.DataFrame(data=daily_data)

        return dataframe

    def read_stream(self):
        """
        Raises:
            NotImplementedError: Reading from the Open Meteo API is not supported
        """
        raise NotImplementedError(
            "Reading from the Open Meteo API is not supported"
        )


class MultiFileSource(SourceInterface):

    def __init__(
            self,
            file_confs: List[Tuple[str, str]],
    ):
        # TODO: doc
        self.file_confs = file_confs


    @staticmethod
    def system_type():
        """
        Attributes:
            SystemType (Environment): Requires PYTHON
        """
        return SystemType.PYTHON

    @staticmethod
    def libraries():
        libraries = Libraries()
        return libraries

    @staticmethod
    def settings() -> dict:
        return {}

    def pre_read_validation(self):
        return True

    def post_read_validation(self):
        return True

    def read_batch(self) -> List[pd.DataFrame]:
        dfs = []
        for file_path, file_type in self.file_confs:
            if file_type == "parquet":
                df = pd.read_parquet(file_path)
                dfs.append(df)
        return dfs

    def read_stream(self):
        """
        Raises:
            NotImplementedError: Reading from the Open Meteo API is not supported
        """
        raise NotImplementedError(
            "Reading from the Open Meteo API is not supported"
        )


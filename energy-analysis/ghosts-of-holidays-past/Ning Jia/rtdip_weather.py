import pandas as pd
import requests
import json
import logging

from rtdip_sdk.pipelines.sources import SourceInterface
from rtdip_sdk.pipelines._pipeline_utils.models import Libraries, SystemType

class PythonHistoricalWeatherSource(SourceInterface):

    weather_url: str = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, params : dict) -> None:
        if "hourly" not in params.keys():
            params["hourly"] = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",  "surface_pressure"]
        
        self.params = params
        
    @staticmethod
    def system_type():
        """
        Attributes:
            SystemType (Environment): Requires Python
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

    def _fetch_from_url(self, url_suffix: str, params: dict) -> bytes:
        """
        Gets data from external Weather API.

        Args:
            url_suffix: String to be used as suffix to weather url.
            params: Params for the weather url.

        Returns:
            Raw content of the data received.

        """
        url = f"{self.weather_url}{url_suffix}"
        logging.info(f"Requesting URL - `{url}` with params - {params}")

        response = requests.get(url, params)
        code = response.status_code

        if code != 200:
            raise HTTPError(
                f"Unable to access URL `{url}`."
                f"Received status code {code} with message {response.content}"
            )

        return response.content
    
    def read_batch(self) -> pd.DataFrame:
        rename_cols = {
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",   
            "dew_point_2m":"dew_point",
            "wind_speed_10m": "wind_speed",
            "surface_pressure": "pressure",
        }
        
        response = json.loads(
            self._fetch_from_url("", self.params).decode(
                "utf-8"
            )
        )

        df = pd.DataFrame(response["hourly"]).rename(rename_cols, axis=1)
        df['time'] = pd.to_datetime(df['time'])
        return df 

    def read_stream(self):
        """
        Raises:
            NotImplementedError: Open-Meteo source does not support the stream operation.
        """
        raise NotImplementedError(
            "Open-Meteo source does not support the stream operation."
        )

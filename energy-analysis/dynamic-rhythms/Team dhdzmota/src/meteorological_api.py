import geopandas as gpd
import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import requests
import time
import warnings


import src.utils as utils

METEOROLOGICAL_DOWNLOADABLE_URL_KEY = 'METEOROLOGICAL_DOWNLOADABLE'
meaning_dict = {
    'begin_datetime': 'storm_start',
    'end_datetime': 'storm_end',
    'run_start_time_min': 'outage_start',
    'run_start_time_max': 'outage_end'
}
API_CALLS_ON = False
METEOROLOGICAL_API_KEY = 'METEOROLOGICAL_API_URL'
config = utils.get_config_file()  # Bottleneck
parameters = utils.get_parameters_file()  # Bottleneck
RANDOM_SEED = utils.RANDOM_SEED
ELEMENT_NB = 200
SLEEP = 5

GENERAL_PATH = utils.get_general_path()
RAW_DATA_PATH = utils.get_data_path('raw')
INTERIM_DATA_PATH = utils.get_data_path('interim')
EXTERNAL_DATA_PATH = utils.get_data_path('external')
METEOROLOGICAL_DATA_PATH = utils.join_paths(EXTERNAL_DATA_PATH, 'meteorological')
STORM_OUTAGES = utils.join_paths(INTERIM_DATA_PATH, 'storm_outages_2014_2023.parquet')
COUNTY_DATA_PATH = utils.join_paths(EXTERNAL_DATA_PATH, 'county.parquet')

warnings.filterwarnings('ignore')


def get_api_url(lat, lon, datetime_start, datetime_end, temporality='hourly'):
    """

    :param lat: Location Latitude
    :param lon: Location Longitude
    :param datetime_start: Start datetime (yyyy-mm-dd format)
    :param datetime_end: Start datetime (yyyy-mm-dd format)
    :param temporality: Temporality of the meteorological information (eg. Daily, Hourly)
    :return:
        url: Formated string of the API url.
    """
    start = pd.to_datetime(datetime_start).strftime('%Y%m%d')
    end = pd.to_datetime(datetime_end).strftime('%Y%m%d')
    base_url = config[METEOROLOGICAL_API_KEY]
    parameter_list = utils.key_list(parameters)
    temporality_str = f'{temporality}/point?'
    params = f"parameters={(',').join(parameter_list)}"
    community = '&community=RE'
    longitude = f"&longitude={lon}"
    latitude = f"&latitude={lat}"
    start_date = f"&start={start}"
    end_date = f"&end={end}"
    final_format = '&format=JSON'
    url = (f'{base_url}{temporality_str}{params}{community}{longitude}{latitude}'
           f'{start_date}{end_date}{final_format}')
    return url


def get_meteorological_info(url):
    """
    Retrieve and parse meteorological information from a given URL.

    :param url: str
        The URL pointing to the JSON-formatted meteorological data.
    :return: formated_text_information: dict
        Parsed JSON data as a Python dictionary.
    """
    text_information = requests.get(url).text
    formated_text_information = json.loads(text_information)
    time.sleep(1)
    return formated_text_information


def save_meteorological_information(information, path):
    """
    Save meteorological information as a JSON file at the specified path.

    :param information: dict
        The meteorological data to be saved.
    :param path: str
        The file path where the JSON will be stored.
    :return: None
    """
    utils.save_as_json(what=information, where=path)


def get_url_params_from_row_point(storm_outage_row, desired_point=None):
    """
    Generate an API URL for meteorological data based on a row containing storm outage info.

    :param storm_outage_row: pandas Series
        A row from a DataFrame containing storm-related information, including geometry and time range.
    :param desired_point: str, optional
        The column name in the row containing the desired shapely Point (with .x and .y attributes).
    :return: url or np.nan
        URL string to request meteorological data, or np.nan if data already exists.
    """
    lon = storm_outage_row[desired_point].x
    lat = storm_outage_row[desired_point].y
    start = storm_outage_row.one_day_before_storm
    end = storm_outage_row.one_day_after_storm
    identifier = storm_outage_row['episode_fips_id']
    save_path = utils.join_paths(METEOROLOGICAL_DATA_PATH, f'{identifier}.json')
    filexists = utils.check_if_filepath_exists(save_path)
    if not filexists:
        url = get_api_url(lat=lat, lon=lon, datetime_start=start, datetime_end=end)
        return url
    return np.nan


def save_meteorological_info_from_row_point(storm_outage_row_iterated):
    """
    Save meteorological information from a DataFrame row if it is valid and not already saved.

    :param storm_outage_row_iterated: tuple
        A tuple containing the index and a pandas Series representing a storm outage row with meteorological info.
    :return: int or np.nan
        Returns 1 if the file was saved, otherwise np.nan.
    """
    idx, storm_outage_row = storm_outage_row_iterated
    identifier = storm_outage_row['episode_fips_id']
    meteorological_info = storm_outage_row['meteorological_info__cntroid']
    validate = meteorological_info.get('properties')
    save_path = utils.join_paths(METEOROLOGICAL_DATA_PATH, f'{identifier}.json')
    filexists = utils.check_if_filepath_exists(save_path)
    if (not filexists) and (validate is not None):
        save_meteorological_information(information=meteorological_info, path=save_path)
        print(f'Saving info at: {save_path}')
        return 1
    return np.nan


def get_outage_storms():
    """
    Load storm outage data and filter for storms that caused an outage.

    :return: outages_storms: pandas DataFrame
        DataFrame containing only the storms where an outage occurred.
    """
    storm_outages_info = pd.read_parquet(STORM_OUTAGES)
    # We keep only storms that caused an outage.
    outages_storms = storm_outages_info[storm_outages_info.storm_caused_outage == 1]
    return outages_storms


def get_meteorological_storm_fips():
    """
    Retrieve the set of unique storm FIPS identifiers from stored meteorological data files.

    :return: unique_meteorological_storm_fips: set
        Set of filenames corresponding to stored meteorological data, representing storm FIPS identifiers.
    """
    meteorological_files = os.listdir(METEOROLOGICAL_DATA_PATH)
    unique_meteorological_storm_fips = set(meteorological_files)
    return unique_meteorological_storm_fips


def request_json_meteorological_files():
    """
    Download and save missing meteorological JSON files for storms that caused outages.

    This function:
      - Identifies which storm-related meteorological files are missing.
      - Samples a subset of these storms (up to `ELEMENT_NB`) in each iteration.
      - Computes relevant date ranges and geographic centroids.
      - Builds URLs for API requests to fetch meteorological data.
      - Uses multiprocessing to fetch data concurrently.
      - Saves the JSON data to disk.
      - Repeats until no new data is downloaded or a retry limit is reached.

    :return: None
    """
    counties = gpd.read_parquet(COUNTY_DATA_PATH)
    counties['fips_code_id'] = counties['STATEFP'] + counties['COUNTYFP']
    outages_storms = get_outage_storms()

    meteorological_storms_fips = get_meteorological_storm_fips()
    unique_storm_fips = set(outages_storms.episode_fips_id + '.json')

    remaining_to_download = unique_storm_fips - meteorological_storms_fips
    actual_len = len(remaining_to_download)
    len_list = 0
    while actual_len > 0 and len_list < 5:
        outages_storms_candidates = outages_storms[
            (outages_storms.episode_fips_id + '.json').isin(list(remaining_to_download))
        ]
        sample_outages_storms = outages_storms_candidates.sample(ELEMENT_NB)
        sample_outages_storms = sample_outages_storms.rename(columns=meaning_dict)
        sample_outages_storms['one_day_before_storm'] = (
                    sample_outages_storms['storm_start'] - pd.Timedelta(1, unit='D')
        ).dt.date
        sample_outages_storms['one_day_after_storm'] = (
                    sample_outages_storms['storm_end'] + pd.Timedelta(1, unit='D')
        ).dt.date
        sample_outages_storms = sample_outages_storms.merge(counties[['fips_code_id', 'geometry']], on='fips_code_id')
        sample_outages_storms = gpd.GeoDataFrame(sample_outages_storms)
        sample_outages_storms['cntroid'] = sample_outages_storms.geometry.centroid
        sample_outages_storms['url__cntroid'] = sample_outages_storms.apply(
            get_url_params_from_row_point,
            desired_point='cntroid',
            axis=1
        )
        sample_outages_storms = sample_outages_storms[sample_outages_storms['url__cntroid'].notna()]
        t0 = time.perf_counter()
        cpu_counts = mp.cpu_count()
        print(f'Available CPUs: {cpu_counts}')
        with mp.Pool(cpu_counts) as pool:
            sample_outages_storms['meteorological_info__cntroid'] = pool.map(
                get_meteorological_info, sample_outages_storms.url__cntroid
            )
        t1 = time.perf_counter()
        for row in sample_outages_storms.iterrows():
            save_meteorological_info_from_row_point(row)
        print('Done with saving.')
        print(f'Time it took {t1-t0}.')
        time.sleep(SLEEP)
        meteorological_storms_fips = get_meteorological_storm_fips()
        unique_storm_fips = set(outages_storms.episode_fips_id + '.json')
        current_len = actual_len
        remaining_to_download = unique_storm_fips - meteorological_storms_fips
        actual_len = len(remaining_to_download)
        print(f'Previous remaining_to_download number was: {current_len}')
        print(f'Actual remaining_to_download number is: {actual_len}')
        if current_len == actual_len:
            len_list += 1
        else:
            len_list = 0


def getting_downloadable_meteorological_folder():
    """
    Download and extract a folder of meteorological JSON files from a configured URL.

    :return: None
    """
    url = config[METEOROLOGICAL_DOWNLOADABLE_URL_KEY]
    print(f'Getting info from {url}...')
    utils.save_json_from_url_zip_parallel(
        url=url, save_data_path=EXTERNAL_DATA_PATH
    )
    return None


def api_call_is_off():
    """
    Handle the case when API calls are disabled.

    Prints warnings to the user and attempts to retrieve meteorological data
    from an already uploaded file if it doesn't exist locally.

    :return: meteorological_filepath_exists: bool
        True if the meteorological data path exists, False otherwise.
    """
    print(
        f'API calls are turned off. '
        f'If you know what you are doing, change the API_CALLS_ON parameter on the script. '
        f'Otherwise, information might already be existing at {METEOROLOGICAL_DATA_PATH}'
    )
    print('Checking if meteorological information is downloaded...')
    meteorological_filepath_exists = utils.check_if_filepath_exists(METEOROLOGICAL_DATA_PATH)
    if not meteorological_filepath_exists:
        print('Attempting to download from uploaded file...')
        getting_downloadable_meteorological_folder()
    else:
        print(f'Meteorological data exists at {METEOROLOGICAL_DATA_PATH}')
    return meteorological_filepath_exists


def download_process():
    """
    Manage the meteorological data download process based on API call settings.

    If API calls are turned off, it checks for existing data or attempts to download from a shared file.
    If API calls are enabled, it triggers the process to request JSON files from the API.

    :return: None
    """
    if not API_CALLS_ON:
        api_call_is_off()
        return None
    else:
        request_json_meteorological_files()


if __name__ == '__main__':
    download_process()

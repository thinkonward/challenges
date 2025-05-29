'''
This main script looks to download from their corresponding links the CITY and COUNTY dataframes.
'''

import src.utils as utils

CITY_URL_KEY = 'CITY_URL_SHAPEFILE'
COUNTY_URL_KEY = 'COUNTY_URL_SHAPEFILE'
WGS84 = "EPSG:4326"  # World Geodetic System 1984 ensemble

EXTERNAL_DATA_PATH = utils.get_data_path('external')
GENERAL_PATH = utils.get_general_path()
RAW_CITY_DATA_PATH = utils.join_paths(EXTERNAL_DATA_PATH, 'cities_raw')
RAW_COUNTY_DATA_PATH = utils.join_paths(EXTERNAL_DATA_PATH, 'counties_raw')
CITY_DATA_PATH = utils.join_paths(EXTERNAL_DATA_PATH, 'city.parquet')
COUNTY_DATA_PATH = utils.join_paths(EXTERNAL_DATA_PATH, 'county.parquet')


def download_cities():
    """
    Download and save shapefile data for cities from a configured URL.

    :return: None
    """
    config = utils.get_config_file()
    url = config[CITY_URL_KEY]
    main_file = utils.save_shapefile_from_url_zip(
        url=url, save_data_path=RAW_CITY_DATA_PATH
    )
    utils.save_info(main_file, filepath=CITY_DATA_PATH)
    return None


def download_counties():
    """
    Download and save shapefile data for counties from a configured URL.

    :return: None
    """
    config = utils.get_config_file()
    url = config[COUNTY_URL_KEY]
    main_file = utils.save_shapefile_from_url_zip(
        url=url, save_data_path=RAW_COUNTY_DATA_PATH
    )
    utils.save_info(main_file, filepath=COUNTY_DATA_PATH)
    return None


def download():
    """
    Trigger the download process for required geographic data files.

    Currently calls the function to download county-level shapefiles.

    :return: None
    """
    download_counties()


if __name__ == "__main__":
    download()

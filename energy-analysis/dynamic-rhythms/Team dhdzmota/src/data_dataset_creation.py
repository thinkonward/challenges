'''
This script is used to create a dataset based on storms outages and in the meteorological information.
The script should yield the meteorological_data_with_outages.parquet file.
'''
import src.utils as utils

import json
import os
import pandas as pd
import warnings

HOURS_BEFORE_START = 10

GENERAL_PATH = utils.get_general_path()
RAW_DATA_PATH = utils.get_data_path('raw')
INTERIM_DATA_PATH = utils.get_data_path('interim')
EXTERNAL_DATA_PATH = utils.get_data_path('external')
METEOROLOGICAL_DATA_PATH = utils.join_paths(EXTERNAL_DATA_PATH, 'meteorological')
STORM_OUTAGES = utils.join_paths(INTERIM_DATA_PATH, 'storm_outages_2014_2023.parquet')
METEOROLOGICAL_OUTAGES = utils.join_paths(INTERIM_DATA_PATH, 'meteorological_data_with_outages.parquet')

warnings.filterwarnings('ignore')


def process_storm_outages():
    """ Function that process storm_outages (reads file, groups values, and aggregates info).

    :return:
    """
    print('Processing storm-outage data...')
    storm_outages = pd.read_parquet(STORM_OUTAGES)
    storm_outages_grouped = storm_outages.groupby('episode_fips_id').agg(
        EPISODE_ID=('EPISODE_ID', 'first'),
        fips_code_id=('fips_code_id', 'first'),
        episode_description=('episode_description', 'first'),
        begin_datetime=('begin_datetime', 'first'),
        end_datetime=('end_datetime', 'last'),
        storm_duration=('storm_duration', 'first'),
        storm_caused_outage=('storm_caused_outage', 'max'),
        outage_index_id=('outage_index_id', 'first'),
        outage_duration=('outage_duration', 'sum'),
        run_start_time_min=('run_start_time_min', 'min'),
        run_start_time_max=('run_start_time_max', 'max'),
        total_customers_out=('total_customers_out', 'mean'),
    ).reset_index()
    return storm_outages_grouped


def get_data_with_response_variable(met_file, storm_outages_g):
    """ Reads meteorological file from meteorological folder, process it and joins it with storm outages.

    :param met_file: str Text that contains the episode_fips_id.json files.
    :param storm_outages_g:
    :return:
    """
    episode_fips_id = met_file.split('.')[0]
    path = utils.join_paths(METEOROLOGICAL_DATA_PATH, met_file)
    with open(path, 'r') as f:
        data_raw = json.load(f)
    try:
        # Get json information:
        data_parameters = data_raw.get('properties').get('parameter')
        coord0, coord1, coord2 = data_raw.get('geometry').get('coordinates')
        # Make a dataframe to capture information
        data = pd.DataFrame(data_parameters)
        data.index.rename('time', inplace=True)
        data.reset_index(inplace=True)
        data['episode_fips_id'] = episode_fips_id
        data['meteorological_current_datetime_val'] = pd.to_datetime(
            data['time'].apply(lambda x: x[0:4]) +
            '-' +
            data['time'].apply(lambda x: x[4:6]) +
            '-' +
            data['time'].apply(lambda x: x[6:8]) +
            ' ' +
            data['time'].apply(lambda x: x[8:10]) +
            ':00:00'
        )
        data['day_of_year'] = data.meteorological_current_datetime_val.dt.day_of_year
        data['hour_of_day'] = data.meteorological_current_datetime_val.dt.hour
        data['day_of_week'] = data.meteorological_current_datetime_val.dt.day_of_week
        data['month_of_year'] = data.meteorological_current_datetime_val.dt.month
        data['coord0'] = coord0
        data['coord1'] = coord1
        data['coord2'] = coord2
        data['meteorological_nextHour_datetime_val'] = (
                data['meteorological_current_datetime_val'] +
                pd.Timedelta(value=1, unit='hours')
        )
        # Get the response var dataframe
        storm_outage_episode = storm_outages_g[
            storm_outages_g.episode_fips_id == episode_fips_id
        ]
        meaning_dict = {
            'begin_datetime': 'storm_start',
            'end_datetime': 'storm_end',
            'run_start_time_min': 'outage_start',
            'run_start_time_max': 'outage_end'
        }
        storm_outage_episode.rename(columns=meaning_dict, inplace=True)
        storm_outage_episode_columns = [
            'episode_fips_id',
            'storm_start',
            'outage_start',
            'outage_end',
            'storm_duration',
            'total_customers_out',
        ]
        # Join information to get the data with response variable (data_rv)
        data_rv = data.merge(
            storm_outage_episode[storm_outage_episode_columns],
            on='episode_fips_id',
            how='left'
        )
        # We are looking for meteorological information before the outage beginnig.
        data_rv = data_rv[
            data_rv.meteorological_current_datetime_val <= data_rv.outage_start
        ]
        # We will compute features 3 hours before the storm start.
        hours_before_start = (
                data_rv.storm_start - pd.Timedelta(value=HOURS_BEFORE_START, unit='hours')
        )
        data_rv = data_rv[data_rv.meteorological_current_datetime_val >= hours_before_start]

        # We can compute the time it takes for a given meteorological condition's time to the outage
        data_rv['hours_to_outage'] = (
            data_rv.outage_start - data_rv.meteorological_current_datetime_val
        ).dt.total_seconds() / 3600

        # This is the target variable:
        # Will an outage happen in the next hour given the current meteorological conditions?
        data_rv['outage_in_an_hour'] = (data_rv['hours_to_outage'] <= 1).astype(int)
        data_rv['episode_fips_time_id'] = data_rv.episode_fips_id + '_' + data_rv.time.astype(str)
        data_rv.set_index('episode_fips_time_id', inplace=True)
        return data_rv
    except:
        # If there is any error while computing the dataframe, lets simplify and return None.
        print(f'There was an error with {episode_fips_id} at path: {path}')
        return None


def create_outage_meteorological_dataset(save=True):
    """Creates outage meteorological dataset, with the save (if desired).

    :param save: bool
    :return:
    """
    storm_outages_grouped = process_storm_outages()
    meteorological_files = os.listdir(METEOROLOGICAL_DATA_PATH)
    data_with_response_variable = []
    print('Processing meteorological jsons data...')
    for met_file in meteorological_files:
        dwrv = get_data_with_response_variable(
            met_file,
            storm_outages_g=storm_outages_grouped
        )
        data_with_response_variable.append(dwrv)

    cleaned_data_wrv = [
        dwrv for dwrv in data_with_response_variable
        if dwrv is not None
    ]
    outage_meteorological_dataset = pd.concat(cleaned_data_wrv)
    print('Done with process...')
    if save:
        dropcols = [
            'meteorological_nextHour_datetime_val',
            'storm_start',
            'outage_start',
            'outage_end',
        ]
        print(f'Saving data at: {METEOROLOGICAL_OUTAGES}')
        outage_meteorological_dataset.drop(
            dropcols, axis=1
        ).to_parquet(METEOROLOGICAL_OUTAGES)

    return outage_meteorological_dataset


def execute(force=False):
    """ Main function that generates the storm outage with meteorological information.

    :return:
    """
    print('Generating Storm-outage data with meteorological information.')
    if not utils.check_if_filepath_exists(METEOROLOGICAL_OUTAGES) or force:
        create_outage_meteorological_dataset()
    else:
        print(f'File already exists, it is located at: {METEOROLOGICAL_OUTAGES}')


def get_data():
    """ Function to get the METEOROLOGICAL_OUTAGES data

    :return: pd.DataFrame
    """
    if utils.check_if_filepath_exists(METEOROLOGICAL_OUTAGES):
        print(f'Reading {METEOROLOGICAL_OUTAGES} file')
        data = pd.read_parquet(METEOROLOGICAL_OUTAGES)
        return data
    print('File does not exist, please compute it.')
    return None
'''
This main script aims to generate the overlap between storms and outages.
'''

import src.utils as utils
import pandas as pd
import warnings

# Min number of customers affected.
CUSTOMERS_OUT_NB = utils.CUSTOMERS_OUT_NB

# Max timelaps that divide outages.
SEPARATION_HOURS = 18
# Represents 15 minutes, in seconds
MIN_OUTAGE_SECONDS = 60 * 15
NB_15_MIN_IN_HOUR = 4
DAYS_AFTER_STORM_THRESHOLD = 1

GENERAL_PATH = utils.get_general_path()
RAW_DATA_PATH = utils.get_data_path('raw')
INTERIM_DATA_PATH = utils.get_data_path('interim')
DYNAMIC_RYTHMS_DATA_PATH = utils.join_paths(RAW_DATA_PATH, 'dynamic-rhythms-train-data', 'data')
EAGLEI_DATA_PATH = utils.join_paths(DYNAMIC_RYTHMS_DATA_PATH, 'eaglei_data')
STORM_EVENTS_CLEANED_PATH = utils.join_paths(INTERIM_DATA_PATH, 'storm_events_cleaned.csv')
STORM_OUTAGES = utils.join_paths(INTERIM_DATA_PATH, 'storm_outages_2014_2023.parquet')

warnings.filterwarnings('ignore')


def get_outages_index(outages_county):
    """
    Index outages into distinct events based on time separation and customer impact.

    This function:
      - Sorts outage records by start time.
      - Filters for outages affecting a minimum number of customers.
      - Computes time gaps between outages.
      - Uses a defined separation threshold to identify new outage events.
      - Assigns a cumulative index to group outages into discrete events.

    :param outages_county: pandas DataFrame
        Outage records for a specific county, including `run_start_time` and `customers_out`.
    :return: outages_county: pandas DataFrame
        Updated DataFrame with a new column `outage_index` indicating event grouping.
    """
    outages_county = outages_county.sort_values("run_start_time")
    # Then we keep only the relevant outage (affecting a high amount of customers)
    outages_county = outages_county[outages_county.customers_out >= CUSTOMERS_OUT_NB]
    # We can define a separation of continuity to "divide" timelapses, in other words, separate outages events.
    # We calculate the difference in seconds of each outage
    outages_county['second_difference'] = outages_county.run_start_time.diff().dt.total_seconds()
    # Each time we find an interval mark greater than the separation time (defined in separation_hours),
    # we identify it as true (1), or false (0).
    separation_hours_seconds = MIN_OUTAGE_SECONDS * NB_15_MIN_IN_HOUR * (SEPARATION_HOURS)
    outages_county['interval_mark'] = (
            outages_county.second_difference.fillna(MIN_OUTAGE_SECONDS) >= separation_hours_seconds
    ).astype(int)
    # then we do the cumulative sum to "generate an index" of same representation.
    outages_county['outage_index'] = outages_county['interval_mark'].cumsum()
    return outages_county


def process_outages(outages):
    """
    Process outage records to create indexed outage events and summarize their characteristics.

    This function:
      - Groups outage data by `sub_general_id` and applies a custom indexing function.
      - Assigns a unique `outage_index_id` based on location and temporal segmentation.
      - Aggregates relevant statistics for each indexed outage event.
      - Computes outage duration and intensity (customers affected per day).
      - Normalizes state names for consistency.

    :param outages: pandas DataFrame
        Raw outage records with fields like `run_start_time`, `customers_out`, `fips_code_id`, etc.
    :return: outages_index_resumed: pandas DataFrame
        Aggregated and cleaned outage data indexed by unique outage events.
    """
    # Get the corresponding index.
    outages_index = outages.groupby('sub_general_id').apply(get_outages_index)
    # Generate an id for each index.
    outages_index['outage_index_id'] = (
            outages_index.fips_code_id + '__' + outages_index.outage_index.astype(str).str.zfill(4)
    )
    # Groupby the index, which depends on separation by time to identify if an outage belongs to the same event or to
    # another one
    outages_index_resumed = outages_index.groupby(
        'outage_index_id'
    ).agg(
        fips_code=('fips_code', 'first'),
        fips_code_id=('fips_code_id', 'first'),
        county=('county', 'first'),
        state=('state', 'first'),
        state_id=('state_id', 'first'),
        total_relevant_registers=('customers_out', 'count'),
        total_customers_out=('customers_out', 'sum'),
        run_start_time_min=('run_start_time', 'min'),
        run_start_time_max=('run_start_time', 'max'),
    ).reset_index()

    outages_index_resumed['run_start_time_max'] = (
            outages_index_resumed['run_start_time_max'] + pd.to_timedelta(900, unit='s')
    )
    outages_index_resumed['outage_duration'] = (
            outages_index_resumed.run_start_time_max - outages_index_resumed.run_start_time_min
    ).dt.total_seconds() / 3600 / 24  # We get the days of time difference
    outages_index_resumed['outage_customers_over_duration'] = (
            outages_index_resumed['total_customers_out'] / outages_index_resumed['outage_duration']
    )
    outages_index_resumed['state'] = outages_index_resumed['state'].str.lower()
    return outages_index_resumed


def process_storm_events(storm_events):
    """
    Process raw storm events DataFrame to generate structured storm episode records by FIPS code.

    This function:
      - Converts raw date and time fields into datetime objects.
      - Computes storm durations in hours.
      - Aggregates storm events into episodes by EPISODE_ID.
      - Explodes the FIPS code list to allow per-county analysis.
      - Generates a unique identifier `episode_fips_id` for each storm-county combination.

    :param storm_events: pandas DataFrame
        Raw storm events data containing temporal fields and location identifiers.
    :return: storms_state_exploded: pandas DataFrame
        DataFrame containing one row per storm episode per FIPS code, including timing and description data.
    """
    begin_datetime = (
        storm_events['BEGIN_YEARMONTH'].astype(str) +
        storm_events['BEGIN_DAY'].astype(str).str.zfill(2) +
        storm_events['BEGIN_TIME'].astype(str).str.zfill(4)
    )

    end_datetime = (
        storm_events['END_YEARMONTH'].astype(str) +
        storm_events['END_DAY'].astype(str).str.zfill(2) +
        storm_events['END_TIME'].astype(str).str.zfill(4)
    )

    storm_events['BEGIN_DATETIME'] = pd.to_datetime(begin_datetime, format='%Y%m%d%H%M')
    storm_events['END_DATETIME'] = pd.to_datetime(end_datetime, format='%Y%m%d%H%M')
    storm_events['DURATION_HOURS'] = (
            (storm_events['END_DATETIME'] - storm_events['BEGIN_DATETIME'])
    ).dt.total_seconds() / 3600

    storm_events['fips_code_id'] = storm_events.new_fips.astype(str).str.zfill(5)

    storm_episodes = storm_events.groupby(
        "EPISODE_ID"
    ).agg(
        nb_events=('EVENT_ID', 'count'),
        affected_states=('STATE', 'unique'),
        affected_states_ids=('STATE_FIPS', 'unique'),
        distinct_events=('EVENT_TYPE', 'unique'),
        fips_only_county_code_id=('CZ_FIPS', 'unique'),
        fips_code_id=('fips_code_id', 'unique'),
        touched_cz_names=('CZ_NAME', 'unique'),
        timezone=('CZ_TIMEZONE', 'unique'),
        episode_description=('EPISODE_NARRATIVE', 'first'),
        begin_datetime=('BEGIN_DATETIME', 'min'),
        end_datetime=('END_DATETIME', 'max'),
    ).reset_index()

    storm_episodes['storm_duration'] = (
                (storm_episodes.end_datetime - storm_episodes.begin_datetime).dt.total_seconds() / 3600
    ).replace(0, 0.01)
    storm_episodes['state'] = storm_episodes.affected_states.apply(lambda x: x[0]).str.lower()
    storm_events['fips_code_id'] = storm_events.new_fips.astype(str).str.zfill(5)
    storms_state_exploded = storm_episodes.explode('fips_code_id')
    storms_state_exploded['episode_fips_id'] = (
            storms_state_exploded.EPISODE_ID.astype(str)
            + '_'
            + storms_state_exploded.fips_code_id.astype(str).str.zfill(5)
    )
    storm_state_exploded_columns = [
        'EPISODE_ID',
        'fips_code_id',
        'episode_description',
        'begin_datetime',
        'end_datetime',
        'storm_duration',
        'episode_fips_id'
    ]
    storms_state_exploded = storms_state_exploded[storm_state_exploded_columns]
    return storms_state_exploded


def combining_outages_and_storms(storms_state_exploded, outages_index_resumed):
    """
    Combine storm and outage datasets and compute whether a storm caused an outage based on temporal conditions.

    :param storms_state_exploded: pandas DataFrame
        DataFrame containing storm event data with FIPS identifiers and timestamps.
    :param outages_index_resumed: pandas DataFrame
        DataFrame containing indexed outage information, including start and end times.
    :return: storms_with_response_var: pandas DataFrame
        Merged DataFrame with an added binary column `storm_caused_outage` indicating if the outage
        can be attributed to a storm, along with additional computed time difference features.
    """
    storms_outages = storms_state_exploded.merge(
        outages_index_resumed,
        on='fips_code_id',
        how='left',
    )
    storms_outages['outage_start_minus_storm_start'] = (
        storms_outages['run_start_time_min'] - storms_outages['begin_datetime']
    ).dt.total_seconds() / 3600 / 24  # This constraint is what defines a "legan" join,

    storms_outages['outage_end_minus_storm_end'] = (
        storms_outages['run_start_time_max'] - storms_outages['end_datetime']
    ).dt.total_seconds() / 3600 / 24

    storms_outages['outage_start_minus_storm_end'] = (
        storms_outages['run_start_time_min'] - storms_outages['end_datetime']
    ).dt.total_seconds() / 3600 / 24

    storms_outages['outage_end_minus_storm_start'] = (
        storms_outages['run_start_time_max'] - storms_outages['begin_datetime']
    ).dt.total_seconds() / 3600 / 24

    storms_outages['storm_caused_outage_cond1'] = (storms_outages.outage_start_minus_storm_start >= 0)
    storms_outages['storm_caused_outage_cond2'] = (storms_outages.outage_start_minus_storm_end <= 0)
    storms_outages['storm_caused_outage_cond3'] = (
        storms_outages.outage_start_minus_storm_end.between(0, DAYS_AFTER_STORM_THRESHOLD)
    )

    storm_outages_conditions = (
            (storms_outages.storm_caused_outage_cond1 & storms_outages.storm_caused_outage_cond2) |
            (storms_outages.storm_caused_outage_cond1 & storms_outages.storm_caused_outage_cond3)
    )

    storms_outages.loc[storm_outages_conditions, 'storm_caused_outage'] = 1
    storms_outages.loc[~storm_outages_conditions, 'storm_caused_outage'] = 0
    storms_outages['episode_fips_id'] = (
            storms_outages.EPISODE_ID.astype(str) + '_' + storms_outages.fips_code_id.astype(str)
    )
    storms_caused_outages = storms_outages[storms_outages.storm_caused_outage == 1]
    storms_with_response_var = storms_state_exploded.merge(
        storms_caused_outages[
            ['storm_caused_outage',
             'episode_fips_id',
             'outage_index_id',
             'outage_start_minus_storm_start',
             'outage_end_minus_storm_end',
             'outage_start_minus_storm_end',
             'outage_end_minus_storm_start',
             'outage_duration',
             'run_start_time_min',
             'run_start_time_max',
             'total_customers_out',
             ]
        ],
        how='left',
        on='episode_fips_id'
    )
    storms_with_response_var.storm_caused_outage = storms_with_response_var.storm_caused_outage.fillna(0)
    return storms_with_response_var


def create_storm_caused_outage():
    """
    Create and save a dataset that identifies whether a storm caused a power outage.

    This function:
      - Loads storm event and outage data.
      - Processes outages and storm data separately.
      - Merges both datasets based on temporal and spatial logic.
      - Computes the target variable (`storm_caused_outage`) using defined conditions.
      - Saves the resulting dataset to a Parquet file.

    :return: None
    """
    # Read data
    storm_events = pd.read_csv(STORM_EVENTS_CLEANED_PATH)
    outages = utils.get_required_outages_dfs(
        2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023,
        eaglei_data_path=EAGLEI_DATA_PATH
    )
    # Process outage info
    print('Processing Outages...')
    outages_index_resumed = process_outages(outages=outages)
    # Process storms info
    print('Processing Storm events...')
    storms_state_exploded = process_storm_events(storm_events=storm_events)
    # Merge outages and storms, generate a dataframe if storm caused outage.
    print('Processing Merging Storm events and outages...')
    storms_with_response_var = combining_outages_and_storms(
        storms_state_exploded=storms_state_exploded,
        outages_index_resumed=outages_index_resumed,
    )
    print(f'Saving results into the following path: {STORM_OUTAGES}')
    storms_with_response_var.to_parquet(STORM_OUTAGES)


def execute():
    """
    Execute the pipeline to create the storm-caused outage dataset if it doesn't already exist.

    :return: None
    """
    if not utils.check_if_filepath_exists(STORM_OUTAGES):
        create_storm_caused_outage()
    else:
        print(f'File already exists, it is located at: {STORM_OUTAGES}')


if __name__ == "__main__":
    execute()

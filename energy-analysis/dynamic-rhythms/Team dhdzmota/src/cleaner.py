'''
This main script looks to generate a Cleaned STORM EVENT
dataframe, with corrected fips.
'''

import src.utils as utils
import geopandas as gpd
import pandas as pd
import warnings


from difflib import SequenceMatcher


GENERAL_PATH = utils.get_general_path()
EXTERNAL_DATA_PATH = utils.get_data_path('external')
RAW_DATA_PATH = utils.get_data_path('raw')
INTERIM_DATA_PATH = utils.get_data_path('interim')
COUNTY_DATA_PATH = utils.join_paths(EXTERNAL_DATA_PATH, 'county.parquet')
DYNAMIC_RYTHMS_DATA_PATH = utils.join_paths(
    RAW_DATA_PATH, 'dynamic-rhythms-train-data', 'data'
)
EAGLEI_DATA_PATH = utils.join_paths(DYNAMIC_RYTHMS_DATA_PATH, 'eaglei_data')
NOAA_DATA_PATH = utils.join_paths(DYNAMIC_RYTHMS_DATA_PATH, 'NOAA_StormEvents')
STORM_EVENTS_PATH = utils.join_paths(
    NOAA_DATA_PATH, 'StormEvents_2014_2024.csv'
)
STORM_EVENTS_CLEANED_PATH = utils.join_paths(
    INTERIM_DATA_PATH, 'storm_events_cleaned.csv'
)


WORD_SIMILARITY_THRESHOLD = 0.8


warnings.filterwarnings('ignore')


def word_similarity(w1, w2):
    """
    Get the similarity between w1 and w2 with sequenceMatcher ratio
    which is similar to get a levenshtein distance but normalized.
    :param w1: string of Word1
    :param w2: string of Word2
    :return: similarity: float
        A coefficient between 0 and 1 yielding the similarity between words.
    """
    similarity = SequenceMatcher(None, w1, w2).ratio()
    return similarity


def fix_fips_codes():
    """Function that makes corrections in the storm_events fips by
    reading the information in official county data.
    """
    # Read data
    county = gpd.read_parquet(COUNTY_DATA_PATH)
    storm_events = pd.read_csv(STORM_EVENTS_PATH)
    # Get the corresponding FIPS for county, outages and events.
    county['fips'] = county.GEOID
    fips_code__county_unique = county['fips'].unique()
    storm_events['fips'] = (
            storm_events.STATE_FIPS.astype(str).str.zfill(2) +
            storm_events.CZ_FIPS.astype(str).str.zfill(3)
    )
    fips_code__storm_events_unique = storm_events['fips'].unique()
    # Calculate unofficial fips on dataframe.
    storm_events_unofficial_fips = len(
        set(fips_code__storm_events_unique) -
        set(fips_code__county_unique)
    )
    print(f'Storm Events dataframe contains '
          f'{storm_events_unofficial_fips} unofficial fips.')
    print(f'Assumption: These {storm_events_unofficial_fips} '
          f'unofficial fips might be typos or mistakes.')

    county_columns = ['STATEFP', 'fips', 'NAME']
    county_sol = county[county_columns].drop_duplicates()

    se_columns = ['STATE_FIPS', 'fips', 'CZ_NAME']
    storm_sol = storm_events[se_columns].drop_duplicates()

    storm_sol['STATEFP'] = storm_sol['STATE_FIPS'].astype(str).str.zfill(2)
    storm_sol.drop('STATE_FIPS', axis=1, inplace=True)
    solved_storm_sol = []
    # For each state in county, we are analyzing the FIPS in
    # one and another dataframe.
    for state in county_sol.STATEFP.unique():
        county_sol_state = county_sol[county_sol.STATEFP == state]
        storm_sol_state = storm_sol[storm_sol.STATEFP == state]
        official_fips = county_sol_state.set_index('fips').NAME.to_dict()
        if storm_sol_state.shape[0] > 0:
            for fips_id, name in official_fips.items():
                name_condition1 = (storm_sol_state.CZ_NAME.str.lower().str
                                   .contains(name.lower()))
                name_condition2 = storm_sol_state.CZ_NAME.str.lower().apply(
                    lambda x: word_similarity(x, name)
                ) > WORD_SIMILARITY_THRESHOLD
                storm_sol_state.loc[
                    ((name_condition1) | (name_condition2)), 'new_fips'
                ] = fips_id
            solved_storm_sol.append(storm_sol_state)
    solved_storm_sol_df = pd.concat(solved_storm_sol)
    solved_storm_sol_df['fips__cz_name'] = (
            solved_storm_sol_df.fips + solved_storm_sol_df.CZ_NAME
    )
    storm_events['fips__cz_name'] = storm_events.fips + storm_events.CZ_NAME
    storm_events_new_fips = storm_events.merge(
        solved_storm_sol_df[['fips__cz_name', 'new_fips']],
        on='fips__cz_name',
        how='left'
    )
    storm_events_new_fips.loc[
        storm_events_new_fips.new_fips.isna(), 'new_fips'
    ] = storm_events_new_fips.loc[storm_events_new_fips.new_fips.isna()].fips
    fips_code__storm_events_new_unique = storm_events_new_fips[
        'new_fips'
    ].unique()
    storm_events_unofficial_fips_new = len(
        set(fips_code__storm_events_new_unique) - set(fips_code__county_unique)
    )
    print(f'After cleaning, Storm Events dataframe contains'
          f' {storm_events_unofficial_fips_new} unofficial fips.')
    corrected_fips = (
            storm_events_unofficial_fips - storm_events_unofficial_fips_new
    )
    print(f'Total amount of corrected fips: {corrected_fips}')
    print(f'Saving Storm events cleaned dataframe '
          f'at: {STORM_EVENTS_CLEANED_PATH}')
    storm_events_new_fips.to_csv(STORM_EVENTS_CLEANED_PATH)
    corrected_percentage = corrected_fips/storm_events_unofficial_fips * 100
    print(f'It was possible a correction of'
          f' about {corrected_percentage}% of cases.')


def execute():
    """
    Executes the preprocessing pipeline for storm events data.

    If the cleaned storm events file does not exist, it triggers the `fix_fips_codes`
    function to generate it. Otherwise, it informs the user that the file already exists.

    :return: None
    """
    if not utils.check_if_filepath_exists(STORM_EVENTS_CLEANED_PATH):
        fix_fips_codes()
    else:
        print(f'File already exists, '
              f'it is located at: {STORM_EVENTS_CLEANED_PATH}')


if __name__ == "__main__":
    execute()

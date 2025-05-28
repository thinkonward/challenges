import pandas as pd

from sklearn.model_selection import train_test_split

import src.feature_generation as feature_generation
import src.utils as utils


RANDOM_SEED = utils.RANDOM_SEED
TRAIN_WINDOW_DATA_PERCENTAGE = 0.8
OOT_WINDOW_DATA_PERCENTAGE = 0.15

SPLITTED_FILENAMES = ['train', 'test', 'eval', 'cal', 'OOT']

INTERIM_DATA_PATH = utils.get_data_path('interim')

# Splits train 80 and test 20
TEST_SIZE = 0.2
# From the remaining train, splits into a transitory set,
# which will be converted into cal and eval sets
TRANS_SIZE = 0.2
# Split of the eval and cal to be CAL_EVAL_SIZE  and 1-CAL_EVAL_SIZE,
# corresponding to TRANSIZE/2 form the remining train
CAL_EVAL_SIZE = 0.5


def get_train_and_oot_candidates():
    # Read data
    data = feature_generation.get_data()
    print('Getting train and OOT windows...')
    # At an episode_fips_id granularity, get the max and min dates.
    episode_fips_dates = data.groupby('episode_fips_id').agg(
        min_date=('meteorological_current_datetime_val', 'min'),
        max_date=('meteorological_current_datetime_val', 'max')
    )
    # See corresponding percentages through time.
    episode_fips_dates_max = episode_fips_dates.max_date.value_counts(
        normalize=True
    ).sort_index().cumsum()
    episode_fips_dates_min = episode_fips_dates.min_date.value_counts(
        normalize=True
    ).sort_index().cumsum()
    # Get dates to split temporal data at a desired corresponding percentage.
    train_perc = TRAIN_WINDOW_DATA_PERCENTAGE*100
    oot_perc = OOT_WINDOW_DATA_PERCENTAGE*100
    print(f'Train window data has {train_perc}% of data.')
    print(f'OOT window data has {oot_perc}% of data.')
    dis_data = (1-TRAIN_WINDOW_DATA_PERCENTAGE-OOT_WINDOW_DATA_PERCENTAGE)*100
    print(f'We are discarding {dis_data}% of data.')
    max_train_date = episode_fips_dates_max[
        episode_fips_dates_max < TRAIN_WINDOW_DATA_PERCENTAGE
    ].tail(1).index[0]
    min_oot_date = episode_fips_dates_min[
        episode_fips_dates_min > (1-OOT_WINDOW_DATA_PERCENTAGE)
    ].head(1).index[0]
    # Split the dataframe, get the corresponding episode_fips ids.
    train_window_candidates = episode_fips_dates[
        episode_fips_dates.max_date < max_train_date
    ].index
    oot_window_candidates = episode_fips_dates[
        episode_fips_dates.max_date > min_oot_date
    ].index
    print('Done.')
    return data, train_window_candidates, oot_window_candidates


def train_test_cal_eval_splits(
        train_window_candidates, oot_window_candidates, print_percentages=True
):
    print('Splitting datasets')
    print('Getting test dataset...')
    train_episode_fips_id, test_episode_fips_id = train_test_split(
        train_window_candidates, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print('Getting train dataset...')
    train_episode_fips_id, transitory_episode_fips_id = train_test_split(
        train_episode_fips_id, test_size=TRANS_SIZE, random_state=RANDOM_SEED
    )
    print('Getting cal and eval datasets...')
    cal_episode_fips_id, eval_episode_fips_id = train_test_split(
        transitory_episode_fips_id,
        test_size=CAL_EVAL_SIZE,
        random_state=RANDOM_SEED
    )
    print('Done.')
    datasets_fips_ids = {
        'train': train_episode_fips_id,
        'test': test_episode_fips_id,
        'cal': cal_episode_fips_id,
        'eval': eval_episode_fips_id,
        'oot': oot_window_candidates
    }
    if print_percentages:
        datasets_fips_ids_len = {
            'train': len(train_episode_fips_id),
            'test': len(test_episode_fips_id),
            'cal': len(cal_episode_fips_id),
            'eval': len(eval_episode_fips_id),
            'oot': len(oot_window_candidates)
        }
        tot_len = sum([v for v in datasets_fips_ids_len.values()])
        print('Datasets relative percentages:')
        for name, dataset_len in datasets_fips_ids_len.items():
            print(f'    {name}: {dataset_len/tot_len*100}%')
    return datasets_fips_ids


def get_datasets(data, datasets_fips_ids, save=True):
    for dataset_name, dataset_fips_id in datasets_fips_ids.items():
        dataset = data[data.episode_fips_id.isin(dataset_fips_id)]
        if save:
            save_individual_dataset(name=dataset_name, data=dataset)



def read_datasets():
    checking_vals = {}
    for name in SPLITTED_FILENAMES:
        filename = utils.join_paths(INTERIM_DATA_PATH, f'{name}.parquet')
        checking_vals[name] = utils.check_if_filepath_exists(filename)
    if not all(checking_vals.values()):
        print('A dataset does not exist...')
        print(checking_vals)
        print('Execute the splitting_process function.')
        return None
    datasets = {}
    for name in checking_vals.keys():
        filename = utils.join_paths(INTERIM_DATA_PATH, f'{name}.parquet')
        print(f'Reading data for {name} at {filename}.')
        datasets[name] = pd.read_parquet(filename)
    return datasets


def save_individual_dataset(name, data):
    filepath = utils.join_paths(INTERIM_DATA_PATH, f'{name}.parquet')
    print(f'Saving {name} data into the path {filepath}')
    data.to_parquet(filepath)



def save_datasets(datasets):
    for name, data in datasets.items():
        save_individual_dataset(name, data)


def splitting_process():
    data, train_window_candidates, oot_window_candidates = (
        get_train_and_oot_candidates()
    )
    datasets_fips_ids = train_test_cal_eval_splits(
        train_window_candidates, oot_window_candidates
    )
    get_datasets(data, datasets_fips_ids)


if __name__ == "__main__":
    splitting_process()

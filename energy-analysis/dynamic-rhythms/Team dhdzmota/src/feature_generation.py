import pandas as pd

from pandas import DataFrame

import src.data_dataset_creation as data_dataset_creation
import src.utils as utils


RANGE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

INTERIM_DATA_PATH = utils.get_data_path('interim')
OUTAGE_FEATURES_FILE = utils.join_paths(INTERIM_DATA_PATH, 'outage_features.parquet')


class OutageFeatures:
    """
     Creates all features for outage Model.
    """
    def __init__(self, data: DataFrame) -> None:
        self.df = data.copy()
        self.no_work_cols = [
            'time',
            'episode_fips_id',
            'meteorological_current_datetime_val',
            'outage_in_an_hour',
            'hours_to_outage',
            'day_of_year',
            'hour_of_day',
            'day_of_week',
            'month_of_year',
            'coord0',
            'coord1',
            'coord2',
            'total_customers_out',
            'storm_duration',
        ]
        self.work_cols = [
            col for col in self.df.columns if col not in self.no_work_cols
        ]

    def get_diff_features(self) -> None:
        """Generates features about:
        - Difference between temperature at 10 meters and at 2 meters.
        - Difference between Wind speed at 50 meters and at 2 meters.

        :return:
        """
        self.df['diff_between_t10m_t2m'] = self.df['T10M'] - self.df['T2M']
        self.df['diff_between_t50m_t2m'] = self.df['WS50M'] - self.df['WS2M']
        self.df['diff_between_ALLSKY_SFC_DWN_SW_LW'] = self.df['ALLSKY_SFC_SW_DWN'] - self.df['ALLSKY_SFC_LW_DWN']
        self.df['diff_between_CLRSKY_SFC_DWN_SW_LW'] = self.df['CLRSKY_SFC_SW_DWN'] - self.df['CLRSKY_SFC_LW_DWN']
        self.df['diff_between_CLRSKY_ALLSKY_SFC_LW_DWN'] = self.df['CLRSKY_SFC_LW_DWN'] - self.df['ALLSKY_SFC_LW_DWN']
        self.df['diff_between_CLRSKY_ALLSKY_SFC_SW_DWN'] = self.df['CLRSKY_SFC_SW_DWN'] - self.df['ALLSKY_SFC_SW_DWN']
        #self.work_cols += [
        #    'diff_between_t10m_t2m',
        #    'diff_between_t50m_t2m',
        #    'diff_between_ALLSKY_SFC_DWN_SW_LW',
        #    'diff_between_CLRSKY_SFC_DWN_SW_LW',
        #    'diff_between_CLRSKY_ALLSKY_SFC_LW_DWN',
        #    'diff_between_CLRSKY_ALLSKY_SFC_SW_DWN',
        #]

    def get_feature_previous_n_hours(self, col: str, n: int) -> None:
        """For a given column, it generates a new column with the values for the previous n hours.

        :param col: str Column to be calculated
        :param n: Hours back to get the value
        :return:
        """
        self.df[f'{col}_{n}_hours_ago'] = self.df.groupby('episode_fips_id')[col].shift(n)

    def get_delta_featues(self, col: str) -> None:
        """For a given column it calculates:
        - Diference between the current value vs the value n hours before (in this case 1, 2 and 3 hours)
        - Diference between value hours (i.e. the value 1 hour before vs the value 3 hours before)

        :param col: str Column to calculate the deltas
        :return:
        """
        for ix in RANGE:
            self.df[f'{col}_delta_{ix}_hour'] = self.df[col] - self.df[f'{col}_{ix}_hours_ago']
        for ix in RANGE[:-1]:
            self.df[f'{col}_delta_{ix}_previous'] = (
                    self.df[f'{col}_{ix}_hours_ago'] - self.df[f'{col}_{ix+1}_hours_ago']
            )

    def get_tendency_features(self, col: str) -> None:
        """Calculate the tendency of the values with the follow logic:
        - if the value increase the tendency will be equal to 1
        - if the value mantains the tendency will be equal to 0
        - if the value decrease the tendency will be equal to 0
        :param col: str Column to calculate features.
        :return:
        """
        def tendency_func(x):
            return 1 if x > 0 else (-1 if x < 0 else 0)

        self.df[f'{col}_previous_tendency'] = self.df[f'{col}_delta_previous'].apply(tendency_func)
        self.df[f'{col}_two_previous_tendency'] = self.df[f'{col}_delta_two_previous'].apply(tendency_func)
        self.df[f'{col}_current_tendency'] = self.df[f'{col}_delta_one_hour'].apply(tendency_func)

    def drop_null_rows_from_lag(self) -> None:
        """ Drops the Null rows that are naturally generated due to the lag.
        :return:
        """
        self.df.dropna(subset=[f'T2M_{max(RANGE)}_hours_ago'], inplace=True)

    def get_features(self) -> DataFrame:
        """Create all features for the outage model.

        :return: DataFrame: dataframe with all the features calculated.
        """
        self.df.sort_values(
            by=['episode_fips_id', 'meteorological_current_datetime_val'],
            inplace=True
        )
        self.get_diff_features()
        for col in self.work_cols:
            for ix in RANGE:
                self.get_feature_previous_n_hours(col, ix)
            self.get_delta_featues(col)
        self.drop_null_rows_from_lag()
        return self.df


def compute_features(save=True, return_df=False):
    """
    Compute engineered features from the input dataset and optionally save or return them.

    :param save: bool, optional
        Whether to save the computed features to disk. Default is True.
    :param return_df: bool, optional
        Whether to return the resulting DataFrame. Default is False.
    :return: data_features: pandas DataFrame (optional)
        The computed feature set, only returned if `return_df` is True.
    """
    data = data_dataset_creation.get_data()
    print('Computing the features...')
    outage_features = OutageFeatures(data=data)
    data_features = outage_features.get_features()
    print('Done.')
    if save:
        print(f'Saving feature data at: {OUTAGE_FEATURES_FILE}')
        data_features.to_parquet(OUTAGE_FEATURES_FILE)
    if return_df:
        return data_features


def get_data():
    """ Function to get the METEOROLOGICAL_OUTAGES data

    :return: pd.DataFrame
    """
    if utils.check_if_filepath_exists(OUTAGE_FEATURES_FILE):
        print(f'Reading {OUTAGE_FEATURES_FILE} file')
        data = pd.read_parquet(OUTAGE_FEATURES_FILE)
        return data
    print('File does not exist, please compute it.')
    return None


if __name__ == "__main__":
    compute_features(save=True, return_df=False)

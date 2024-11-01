import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.fft import fft

def interquantiles_80_20(x):
    return x.quantile(0.8) - x.quantile(0.2)

# ====================================================================================================
# FUNCTIONS TO COMPUTE PERCENTILE IN GROUPBY OPERATIONS

def interquantiles_80_20(x):
    return x.quantile(0.8) - x.quantile(0.2)

def interquantiles_90_10(x):
    return x.quantile(0.9) - x.quantile(0.1)

def interquantiles_75_25(x):
    return x.quantile(0.75) - x.quantile(0.25)

# ====================================================================================================
# DATE FEATURES

def create_date_features(df) :
    """
    Creates several date and time-based features from a timestamp column in the input DataFrame.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame containing a 'timestamp' column of datetime type.
    
    Returns:
    - df (pd.DataFrame): Output DataFrame with newly generated date features.
    
    The following features are created:
    
    Date Features:
    - date: Extracted date (YYYY-MM-DD) from the timestamp.
    - year: Year extracted from the timestamp.
    - month: Month extracted from the timestamp.
    - day: Day of the month extracted from the timestamp.
    - dayofweek: Day of the week (0 = Monday, 6 = Sunday).
    - week: ISO week number of the year.
    - hour: Hour of the day extracted from the timestamp.
    - minute: Minute of the hour extracted from the timestamp.
    - period: Month and day extracted as a string (MM-DD).
    - hour_minute: Time in hours and minutes extracted from the timestamp (HH:MM).
    
    Time Calculations:
    - seconds_since_2018: Total number of seconds since January 1st, 2018, 00:00:00.

    Categorical Features:
    - is_weekend: Binary feature (1 for weekend, 0 for weekdays).
    
    Seasonal Features:
    - season: Season based on the timestamp (0 = Winter, 1 = Spring, 2 = Summer, 3 = Autumn).
      The season is determined by the total seconds since 2018 and custom seasonal ranges.
    
    Period Features:
    - hour_period: Time range buckets for each hour of the day (e.g., '1_6', '7_8', '9_11').
    - day_period: Part of the day corresponding to the hour ('night', 'morning', 'noon', 'afternoon', 'evening').

    Holiday Features:
    - is_holiday: Binary feature indicating whether the date falls on a predefined list of holidays.
    
    Notes:
    - The holidays list is based on fixed holiday dates (MM-DD) and may need adjustment for different regions or years.
    """

    # Date features
    df["date"]        = df['timestamp'].astype(str).apply(lambda x : x[:11])
    df["year"]        = df['timestamp'].dt.year
    df["month"]       = df['timestamp'].dt.month
    df["day"]         = df['timestamp'].dt.day
    df["dayofweek"]   = df['timestamp'].dt.dayofweek
    df["week"]        = df['timestamp'].dt.isocalendar().week
    df["hour"]        = df['timestamp'].dt.hour
    df["minute"]      = df['timestamp'].dt.minute
    df["period"]      = df['timestamp'].astype(str).apply(lambda x : x[5:10])
    df["hour_minute"] = df['timestamp'].astype(str).apply(lambda x : x[11:16])
    
    # Seconds since 2018-01-01 00:00:00
    df['seconds_since_2018'] = (df['timestamp'] - pd.Timestamp('2018-01-01')).dt.total_seconds().astype(int)

    # Week-end
    df['is_weekend'] = df['dayofweek'].apply(lambda x : x in [5, 6]).astype(int)

    # Month features (season)
    def get_season(s):
        """ Create season (int from 0 to 4) based on the number of seconds elapsed since 2018 """ 
        for season, [start, end] in [[0, [0, 2.5]],
                                     [1, [2.5, 4]],
                                     [2, [4, 6]],
                                     [3, [6, 8.5]],
                                     [4, [8.5, 10.5]],
                                     [0, [10.5, 12]],
                                    ] :
            if start*30*24*3600 <= s <= end*30*24*3600 :
                return season
        return 0
    df["season"] = df['seconds_since_2018'].apply(get_season)
                                      
    # Hour period
    def get_hour_period(hour):
        """ Create hour period """
        # Return '22_24' if hour is 0
        if hour == 0 :
            return '22_24'
        # Iterate over predefined hour ranges and return corresponding range as string
        for start, end in [[1, 6],
                           [7, 8],
                           [9, 11],
                           [12, 14],
                           [15, 18],
                           [19, 21],
                           [22, 24],
                          ] :
            if start <= hour <= end :
                return f'{start}_{end}'
    df["hour_period"] = df['hour'].apply(lambda hour : get_hour_period(hour))

    # Day period
    def get_day_period(hour):
        if 1 <= hour <= 6 :
            return 'night'
        elif 7 <= hour <= 11 :
            return 'morning'
        elif 11 <= hour <= 14 :
            return 'noon'
        elif 14 <= hour <= 19 :
            return 'afternoon'
        elif 19 <= hour <= 24 or hour==0 :
            return 'evening'
    df["day_period"] = df['hour'].apply(lambda hour : get_day_period(hour))

    # Identify holidays
    holidays = ['01-01',
                '01-15',
                '02-19',
                '05-28',
                '07-04',
                '09-03',
                '10-08',
                '11-11',
                '11-12',
                '11-22',
                '12-25']
    df['is_holiday'] = df['timestamp'].astype(str).apply(lambda x : any(x[5:11].startswith(s) for s in holidays)).astype(int)
    
    # Return
    return df


# ====================================================================================================
# SHIFTED FEATURES


def get_shifted_features(df) :
    
    """
    Generates various energy-related shifted and differential features based on quarter-hour intervals
    in the input DataFrame. These features capture energy changes over time, including normalized differences,
    comparisons to past and future quarters, and identification of peaks in energy consumption.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame containing 'energy' and 'timestamp' columns.

    Returns:
    - df (pd.DataFrame): DataFrame with added features related to energy shifts and differences over time.
    
    The following features are created:

    Normalized Energy Features:
    - normalized_energy: Z-score normalized energy values (mean-centered and divided by std deviation).
    - normalized_energy_diff_from_last_quarter: Difference in normalized energy from the last quarter.
    - normalized_energy_diff_from_last_week: Difference in normalized energy from the same time one week ago.
    - normalized_energy_diff_from_last_month: Difference in normalized energy from the same time one month ago.
    - abs_normalized_energy_diff_from_last_quarter: Absolute difference from last quarter.
    - abs_normalized_energy_diff_from_last_week: Absolute difference from last week.
    - abs_normalized_energy_diff_from_last_month: Absolute difference from last month.
    
    Day Mean Comparison:
    - diff_to_day_mean: Difference between energy and the daily average.
    - diff_to_dayperiod_mean: Difference between energy and the average for the specific day period (e.g., morning, noon).
    
    Quarter and Hour Differences:
    - diff_from_last_quarter: Difference in energy compared to the last quarter.
    - diff_from_next_quarter: Difference in energy compared to the next quarter.
    - diff_from_last_2quarters: Difference in energy compared to two quarters ago.
    - diff_from_next_2quarters: Difference in energy compared to two quarters ahead.
    - diff_from_last_hour: Difference in energy compared to the last hour.
    - diff_from_next_hour: Difference in energy compared to the next hour.
    - diff_from_last_week: Difference in energy compared to the same time one week ago.
    - diff_from_next_week: Difference in energy compared to the same time one week ahead.
    - diff_from_last_month: Difference in energy compared to the same time one month ago.
    - diff_from_next_month: Difference in energy compared to the same time one month ahead.
    - Absolute differences (e.g., abs_diff_from_last_quarter) and percentage differences (e.g., percent_diff_from_last_quarter) are also calculated for each shift.
    
    Binary Change Indicators:
    - energy_has_increased_since_last_quarter: Binary flag indicating whether energy has increased since the last quarter.
    - energy_has_decreased_since_last_quarter: Binary flag indicating whether energy has decreased since the last quarter.
    
    Consecutive Quarter Changes:
    - n_consecutive_quarters_with_increase: Number of consecutive quarters with increasing energy.
    - n_consecutive_quarters_with_decrease: Number of consecutive quarters with decreasing energy.
    
    Comparison to Previous/Next Quarters:
    - is_equal_to_last_or_next_quarter: Binary flag indicating if energy is the same as the previous or next quarter.
    - is_almost_equal_to_last_or_next_quarter: Binary flag indicating if energy is nearly equal to the previous or next quarter (within a small threshold).
    
    Peak Identification:
    - is_peak_smooth_xx: Binary flag indicating whether the quarter represents a smooth peak, where energy is nearly constant (within a certain percentage).
    - is_peak_xx: Binary flag indicating whether the quarter represents a significant peak, where energy increased/decreased sharply (by a certain percentage).
    - is_peak_increase_xx: Binary flag for detecting energy increase peaks (above a certain percentage).
    - is_peak_decrease_xx: Binary flag for detecting energy decrease peaks (below a certain percentage).
    - is_peak_ascent_xx: Binary flag for detecting sharp increase followed by sharp decrease (ascent).
    - is_peak_descent_xx: Binary flag for detecting sharp decrease followed by sharp increase (descent).
    
    Notes:
    - Quarter refers to a 15-minute interval.
    - Time shifts are applied using `.shift()` with various lags (e.g., 1 quarter = 15 minutes, 4 quarters = 1 hour).
    - Peaks are identified based on percentage thresholds (0.02, 0.15, 0.3, 0.45) to detect energy consumption patterns.
    """
    
    # Normalized energy
    df['normalized_energy'] = (df['energy'].clip(0, None) - df['energy'].clip(0, None).mean()) / df['energy'].clip(0, None).std()
    df['normalized_energy_diff_from_last_quarter'] = (df['normalized_energy'] - df['normalized_energy'].shift(1))
    df['normalized_energy_diff_from_last_week']    = (df['normalized_energy'] - df['normalized_energy'].shift(7*24*4))
    df['normalized_energy_diff_from_last_month']   = (df['normalized_energy'] - df['normalized_energy'].shift(28*24*4))
    df['abs_normalized_energy_diff_from_last_quarter'] = abs(df['normalized_energy_diff_from_last_quarter'])
    df['abs_normalized_energy_diff_from_last_week']    = abs(df['normalized_energy_diff_from_last_week'])
    df['abs_normalized_energy_diff_from_last_month']   = abs(df['normalized_energy_diff_from_last_month'])
    
    # Difference to day-mean
    df['diff_to_day_mean'] = df['energy'] -  df.groupby('date')['energy'].transform('mean')
    df['diff_to_dayperiod_mean'] = df['energy'] -  df.groupby(['date', 'day_period'])['energy'].transform('mean')
    
    # Difference from last/next quarters
    for col_name, lag in [['diff_from_last_quarter', 1], # diff from last quarter
                          ['diff_from_next_quarter', -1], # diff from next quarter
                          ['diff_from_last_2quarters', 2], # diff from last 2th quarter
                          ['diff_from_next_2quarters', -2], # diff from next 2th quarter
                          ['diff_from_last_3quarters', 3], # diff from last 3th quarter
                          ['diff_from_next_3quarters', -3], # diff from next 3th quarter
                          ['diff_from_last_hour', 4], # diff from last hour
                          ['diff_from_next_hour', -4], # diff from next hour
                          ['diff_from_last_2hour', 8], # diff from last 2 hour
                          ['diff_from_next_2hour', -8], # diff from next 2 hour
                          ['diff_from_last_week', 7*24*4], # diff from last week
                          ['diff_from_next_week', -7*24*4], # diff from next week
                          ['diff_from_last_month', 28*24*4], # diff from last month
                          ['diff_from_next_month', -28*24*4], # diff from next month
                         ]:
    
        # Get diff
        df[col_name] = (df['energy'] - df['energy'].shift(lag))
        df['abs_' + col_name] = abs(df[col_name])
        df['percent_' + col_name] = df[col_name] / df['energy']
        df['abs_percent_' + col_name] = abs(df['percent_' + col_name])
        
    # Value increased/decreased from last quarter
    df['energy_has_increased_since_last_quarter'] = (df['diff_from_last_quarter'] > 0).astype(int)
    df['energy_has_decreased_since_last_quarter'] = (df['diff_from_last_quarter'] < 0).astype(int)
    
    # Number of consecutive quarters that increased the value
    k = 'energy_has_increased_since_last_quarter'
    df['n_consecutive_quarters_with_increase'] = df.groupby((df[k] != df[k].shift(1)).cumsum())[k].cumsum()
    k = 'energy_has_decreased_since_last_quarter'
    df['n_consecutive_quarters_with_decrease'] = df.groupby((df[k] != df[k].shift(1)).cumsum())[k].cumsum()
    
    # Value equal to previous or next quarter
    df['is_equal_to_last_or_next_quarter'] = (df[['diff_from_last_quarter', 'diff_from_next_quarter']] == 0).any(axis=1).astype(int)
    df['is_almost_equal_to_last_or_next_quarter'] = (df[['diff_from_last_quarter', 'diff_from_next_quarter']] <= 1e-3 + 1e-7).any(axis=1).astype(int)
    
    # Identify pure peaks
    for percent in [0.02, 0.15, 0.3, 0.45] :
        df[f'is_peak_smooth_{int(100*percent)}'] = ((df['abs_percent_diff_from_last_quarter'] <= percent) & (df['abs_percent_diff_from_next_quarter'] <= percent)).astype(int)
        df[f'is_peak_{int(100*percent)}'] = ((df['abs_percent_diff_from_last_quarter'] >= percent) & (df['abs_percent_diff_from_next_quarter'] >= percent)).astype(int)
        df[f'is_peak_increase_{int(100*percent)}'] = ((df['percent_diff_from_last_quarter'] >= percent) & (df['percent_diff_from_next_quarter'] >= percent)).astype(int)
        df[f'is_peak_decrease_{int(100*percent)}'] = ((df['percent_diff_from_last_quarter'] <= -percent) & (df['percent_diff_from_next_quarter'] <= -percent)).astype(int)
        df[f'is_peak_ascent_{int(100*percent)}']   = ((df['percent_diff_from_last_quarter'] >= percent) & (df['percent_diff_from_next_quarter'] <= -percent)).astype(int)
        df[f'is_peak_descent_{int(100*percent)}']  = ((df['percent_diff_from_last_quarter'] <= -percent) & (df['percent_diff_from_next_quarter'] >= percent)).astype(int)
        
    # Return
    return df

# ====================================================================================================
# FEATURES FROM SHIFTED VALUES

def get_features_from_shifted_energy(df) :
    
    """
    Generates a set of features from percentage differences in energy over various time shifts.
    These features count how often the percentage change in energy exceeds or falls below certain thresholds.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing columns for percentage differences between energy values at different time shifts.
    
    Returns:
    - pd.DataFrame: A one-row DataFrame containing the counts of times the percentage differences exceed positive thresholds or fall below negative thresholds.
    """

    # Initialize dictionnary
    data = {}
    
    # Define thresholds
    thresholds = np.concatenate([np.arange(0.02, 0.10001, 0.02), np.arange(0.1, 1.001, 0.05)])
    
    # Compute how many times we exceed some thr
    for col_name in ['percent_diff_from_last_quarter',
                     'percent_diff_from_next_quarter',
                     'percent_diff_from_last_2quarters',
                     'percent_diff_from_next_2quarters',
                     'percent_diff_from_last_3quarters',
                     'percent_diff_from_next_3quarters',
                     'percent_diff_from_last_hour',
                     'percent_diff_from_next_hour',
                     'percent_diff_from_last_2hour',
                     'percent_diff_from_next_2hour',
                     'percent_diff_from_last_week',
                     'percent_diff_from_next_week',
                     'percent_diff_from_last_month',
                     'percent_diff_from_next_month',
                    ]:
        
        # Values
        values = df[col_name].values
        # Complete data
        for thr in thresholds :
            data[f'n_{col_name}_greater_than_positive_{round(100*thr)}%'] = [np.sum(values > thr)] # Greater
            data[f'n_{col_name}_lower_than_negative_{round(100*thr)}%'] = [np.sum(values < -thr)] # Lower
            
    # Return a one-line dataframe
    return pd.DataFrame(data)


# ====================================================================================================
# VALUE COUNTS FEATURES

def get_value_counts_features(df) :
    
    """
    Generates features based on value counts of energy values across different time dimensions 
    (e.g., month, week, day of the week, day period) and specific energy-related statistics 
    (e.g., idle and peak energy).

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing energy values and time-related columns (e.g., month, week, day of the week, day period).
    
    Returns:
    - pd.DataFrame: A one-row DataFrame containing various features derived from value counts of 
      energy values across different time-related dimensions and thresholds.

    Operations:
    
    1. **Value Counts over Time-based Columns**:
        - The function computes value counts of energy values for each unique value in the columns:
          - 'month': Each month of the year.
          - 'week': Week of the year.
          - 'dayofweek': Days of the week (0 to 6).
          - 'day_period': Different parts of the day (e.g., morning, afternoon).
          
        For each of these columns, the following features are created:
        - `most_seen_energy_values_over_<x>_<col>`: The most frequently observed energy value during period `x` in column `col`.
        - `most_seen_energy_times_over_<x>_<col>`: How many times this most seen energy value appears.
        - `how_many_energy_seen_most_times_over_<x>_<col>`: How many different energy values are seen the most number of times.
    
    2. **Idle and Peak Energy Features**:
        - For each time-based column and each of the quantiles (0.04, 0.1, and 0.2), the function calculates:
          - `energy_values_under_<x>_<col>_quantile_<int(100*q)>`: The energy value at quantile `q` for period `x` in column `col`.
          - `n_energies_with_lower_value_than_idle_<x>_<col>_quantile_<int(100*q)>`: Number of energy values below the quantile (considered idle).
          - `energy_values_greater_<x>_<col>_quantile_<int(100*(1-q))>`: The energy value at the upper quantile (considered a peak).
          - `n_energies_with_greater_value_than_peak_<x>_<col>_quantile_<int(100*(1-q))>`: Number of energy values above the upper quantile (considered peak).
    
    3. **Value Counts over Energy**:
        - The function calculates value counts for the 'energy' column across all periods:
          - `most_seen_energy_values`: The most frequently observed energy value overall.
          - `most_seen_energy_times`: How many times this most seen energy value appears.
          - `how_many_energy_seen_most_times`: How many different energy values are seen the most number of times.

    4. **Threshold-Based Features**:
        - For thresholds ranging from 100 to 2000 (with a step of 200), the function calculates:
          - `n_energy_values_seen_at_least_<thr>_times`: Number of energy values that appear at least `thr` times.

    5. **Special Cases (Negative and Zero Energy)**:
        - Counts the number of rows where:
          - `n_rows_with_negative_energy`: Energy is negative (possibly indicating renewables like solar).
          - `n_rows_with_zero_energy`: Energy is zero.
    
    Returns:
    - A one-row DataFrame with all these features.
    """

    
    # Initialize dictionnary
    data = {}
    
    # -----------------------------------------------------------------------------
    # Value_counts over month, day of week and day_period
    
    for col in ['month', 'week', 'dayofweek', 'day_period'] :
        
        # Value counts
        value_counts = df[[col, 'energy']].value_counts().reset_index()
        
        # For each unique value
        for x in sorted(df[col].unique()) :
            
            # Count of the value
            values = value_counts[value_counts[col] == x]

            # Complete data
            count = values['count'].values
            data[f'most_seen_energy_values_over_{x}_{col}'] = [values['energy'].values[0]]
            data[f'most_seen_energy_times_over_{x}_{col}']  = [count[0]]
            data[f'how_many_energy_seen_most_times_over_{x}_{col}']  = [np.sum(count == count.max())]
            
            # Idle and peak energies
            for q in [0.04, 0.1, 0.2] :

                # Idle
                quantile = np.quantile(count, q)
                data[f'energy_values_under_{x}_{col}_quantile_{int(100*q)}'] = [quantile]
                data[f'n_energies_with_lower_value_than_idle_{x}_{col}_quantile_{int(100*q)}'] = [len(count[count <= quantile])]

                # Peak
                quantile = np.quantile(count, 1-q)
                data[f'energy_values_greater_{x}_{col}_quantile_{int(100*(1-q))}'] = [quantile]
                data[f'n_energies_with_greater_value_than_peak_{x}_{col}_quantile_{int(100*(1-q))}'] = [len(count[count >= quantile])]
            
    # -----------------------------------------------------------------------------
    # Value_counts over all energy values
    
    value_counts = df['energy'].value_counts()
    data['most_seen_energy_values'] = [value_counts.index[0]]
    data['most_seen_energy_times']  = [value_counts.values[0]]
    data['how_many_energy_seen_most_times']  = [np.sum(value_counts == value_counts.max())]
    
    # Iterate over several thresholds
    for thr in range(100, 2001, 200) :
        data[f'n_energy_values_seen_at_least_{thr}_times'] = [len(value_counts[value_counts >= thr])]
        
    # Number of rows with negative energy (indicate renewables such as solar panel)
    data['n_rows_with_negative_energy'] = [len(df[df['energy'] < 0])]
    data['n_rows_with_zero_energy']     = [len(df[df['energy'] == 0])]
        
    # Return a one-line dataframe
    return pd.DataFrame(data)


# ====================================================================================================
# FAST FOURIER TRANSFORM

def get_fft_feats(df) :
    """
    Generates Fast Fourier Transform (FFT) features from the 'energy' column, aggregated by various time-related columns. 
    The FFT is computed on both the mean and sum of energy values grouped by the specified time dimensions.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the 'energy' column and time-related columns such as 
      'week', 'dayofweek', 'hour', 'date', 'hour_minute', 'day_period', and 'hour_period'.
    
    Returns:
    - pd.DataFrame: A one-row DataFrame containing FFT features based on the mean and sum of energy values 
      for different time-based groupings.
    """
    
    # Initiate a dictionnary (faster that directly creating column in final_df)
    data = {}
    
    # Compute
    for col in ['week', 'dayofweek', 'hour', 'date', 'hour_minute', 'day_period', 'hour_period'] :
        
        # MEAN
        fft_values = fft(df.groupby(col)['energy'].mean().values)
        data[f"FFT_{col}_mean_length"] = [len(fft_values)]
        for i in range(10) :
            if i < len(fft_values) :
                data[f"FFT_{col}_mean_{i+1}th_value"] = [np.abs(fft_values[i])]
            else :
                data[f"FFT_{col}_mean_{i+1}th_value"] = [-1]
            
        # SUM
        fft_values = fft(df.groupby(col)['energy'].sum().values)
        data[f"FFT_{col}_sum_length"] = [len(fft_values)]
        for i in range(15) :
            if i < len(fft_values) :
                data[f"FFT_{col}_sum_{i+1}th_value"] = [np.abs(fft_values[i])]
            else :
                data[f"FFT_{col}_sum_{i+1}th_value"] = [-1]
                
    # Return a one-line dataframe
    return pd.DataFrame(data)


# ====================================================================================================
# RAMP-UP AND RAMP-DOWN SPEEDS

def get_ramps_speeds(df) :
    """
    Generates ramp-up and ramp-down speed statistics from the 'energy' column and computes load and unload curves.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the 'energy' column and time-related columns such as 
      'month', 'week', 'dayofweek', and 'day_period'.
    
    Returns:
    - pd.DataFrame: A one-row DataFrame containing:
        - Mean and standard deviation of ramp-up and ramp-down speeds.
        - Ramp speed statistics for each unique value in the specified time columns.
        - Load and unload curves at specific points.
    """
    
    # Initialize dictionnary
    data = {}
    
    # Compute diff
    ramp_up_speed   = df['energy'].diff().clip(lower=0)
    ramp_down_speed = df['energy'].diff().clip(upper=0).abs()
    
    # Store the mean
    data['ramp_up_speed_mean']   = [ramp_up_speed.mean()]
    data['ramp_up_speed_std']    = [ramp_up_speed.std()]
    data['ramp_down_speed_mean'] = [ramp_down_speed.mean()]
    data['ramp_down_speed_std']  = [ramp_down_speed.std()]
        
    # -----------------------------------------------------------------------------
    # Get stats over some periods
    
    for col in ['month', 'week', 'dayofweek', 'day_period'] :
        for x in sorted(df[col].unique()) :
        
            # Mask
            mask = (df[col] == x)
            
            # Store information
            data[f'ramp_up_speed_mean_over_{x}_{col}'] = [ramp_up_speed.mean()]
            data[f'ramp_up_speed_std_over_{x}_{col}']  = [ramp_up_speed.std()]
            data[f'ramp_down_speed_mean_over_{x}_{col}'] = [ramp_down_speed.mean()]
            data[f'ramp_down_speed_std_over_{x}_{col}']  = [ramp_down_speed.std()]
            
            # Sorted
            energy_values = df.loc[mask, 'energy'].values
            
            # Load curve (sorted energy values and cumsum)
            sorted_energy = np.sort(energy_values) # Sort energy consumption values
            S = sorted_energy.sum()
            load_curve    = np.cumsum(sorted_energy) / S  # Normalize cumulative sum
            unload_curve  = np.cumsum(sorted_energy[::-1]) / S
            for i in [0, 20, 100, 200, 300] :
                data[f'load_curve_{i}_over_{x}_{col}']   = [load_curve[i]]
                data[f'unload_curve_{i}_over_{x}_{col}'] = [unload_curve[i]]

    # Return a one-line dataframe
    return pd.DataFrame(data)


# ====================================================================================================
# OPENING AND CLOSING TIMES (PART 1/2)


def get_opening_and_closing_times(df) :
    """
    Estimates opening and closing times based on energy differences in specific time periods and seasons, 
    then calculates various statistics for operating hours.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing columns such as 'is_holiday', 'dayofweek', 'hour_minute', 'month', and 'energy'.

    Returns:
    - pd.DataFrame: A one-row DataFrame containing:
        - Estimated opening and closing times for different periods (e.g., winter, summer, all year).
        - Various operating hours estimations.
        - Mean, median, and standard deviation of these estimations.
    """
    
    # Initialize dictionnary
    data = {}
    
    for group_of_months, name_of_months in [[[3, 5, 10], ''],
                                            [[11, 12, 1, 2], '_winter'],
                                            [[6, 7, 8], '_summer'],
                                            [list(range(1, 13)), '_all_year'],
                                           ] :

        # --------------------------------------------------------------------------------------
        # OPENING TIME

        # Opening time occurs between 3h30 and 12h15, so we take this range (on working period)
        mask = (df['is_holiday'] == 0) & (df['dayofweek'].between(0, 3)) & (df['hour_minute'].between('03:00', '12:45')) & (df['month'].isin(group_of_months))
        df_tmp = df.loc[mask, ["timestamp", "hour_minute", "energy", 'diff_from_last_quarter']].reset_index(drop=True)

        # Consider diff to last quarter
        diff = df_tmp.groupby("hour_minute")['diff_from_last_quarter'].mean()

        # Store stats
        data[f'opening_diff_mean{name_of_months}'] = [np.mean(diff)]
        data[f'opening_diff_std{name_of_months}']  = [np.std(diff)]
        for i, k in enumerate(diff.values):
            data[f"opening_diff_{i+3*4}th_quarter{name_of_months}"] = k
        opening_quarter = diff.argmax() + 3*4 # we add 3*4 corresponding to 3h
        data[f'opening_quarter_estimation{name_of_months}'] = [opening_quarter]


        # Consider 4 consecutive diff to last quarter (one hour)
        df_tmp['sum_of_4_last_diff'] = df_tmp['diff_from_last_quarter'].rolling(4).sum()
        opening_quarter_2 = df_tmp.groupby("hour_minute")['sum_of_4_last_diff'].mean().argmax() + 3*4 # we add 3*4 corresponding to 3h
        data[f'opening_quarter_estimation_2{name_of_months}'] = [opening_quarter_2]

        # --------------------------------------------------------------------------------------
        # CLOSING TIME

        # Closing time occurs between 14h00 and 23h45, so we take this range (on working period)
        mask = (df['is_holiday'] == 0) & (df['dayofweek'].between(0, 3)) & (df['hour_minute'].between('14:00', '23:45')) & (df['month'].isin(group_of_months))
        df_tmp = df.loc[mask, ["timestamp", "hour_minute", "energy", 'diff_from_last_quarter']].reset_index(drop=True)

        # Consider diff to last quarter
        diff = df_tmp.groupby("hour_minute")['diff_from_last_quarter'].mean()

        # Store stats
        data[f'closing_diff_mean{name_of_months}'] = [np.mean(diff)]
        data[f'closing_diff_std{name_of_months}']  = [np.std(diff)]
        for i, k in enumerate(diff.values):
            data[f"closing_diff_{i+14*4}th_quarter{name_of_months}"] = k
        closing_quarter = diff.argmin() + 14*4 # we add 14*4 corresponding to 14h
        data[f'closing_quarter_estimation{name_of_months}'] = [closing_quarter]

        # Consider 4 consecutive diff to last quarter (one hour)
        df_tmp['sum_of_4_last_diff'] = df_tmp['diff_from_last_quarter'].rolling(4).sum().shift(-3) # shift 3 rows to consider the decrease to come (after the hour)
        closing_quarter_2 = df_tmp.groupby("hour_minute")['sum_of_4_last_diff'].mean().argmax() + 14*4 # we add 14*4 corresponding to 14h
        data[f'closing_quarter_estimation_2{name_of_months}'] = [closing_quarter_2]

        # --------------------------------------------------------------------------------------
        # OPERATING TIME

        data[f'operating_hours_estimation_1{name_of_months}'] = [(closing_quarter - opening_quarter) / 4]
        data[f'operating_hours_estimation_2{name_of_months}'] = [(closing_quarter_2 - opening_quarter_2) / 4]
        data[f'operating_hours_estimation_3{name_of_months}'] = [((closing_quarter + closing_quarter_2)/2 - (opening_quarter_2+opening_quarter)/2) / 4]
        data[f'operating_hours_estimation_MIN{name_of_months}'] = [(min(closing_quarter, closing_quarter_2) - max(opening_quarter_2, opening_quarter)) / 4]
        data[f'operating_hours_estimation_MAX{name_of_months}'] = [(max(closing_quarter, closing_quarter_2) - min(opening_quarter_2, opening_quarter)) / 4]
        
    # --------------------------------------------------------------------------------------
    # MEAN, MEDIAN and STD of estimations
    
    for prefix in ['opening_quarter_estimation',
                   'closing_quarter_estimation',
                   'operating_hours_estimation',
                   'operating_hours_estimation_1',
                   'operating_hours_estimation_2',
                   'operating_hours_estimation_3',
                   'operating_hours_estimation_MIN',
                   'operating_hours_estimation_MAX',
                  ] :
        values = [data[k] for k in data.keys() if k.startswith(prefix)]
        data[f"MEAN_{prefix}"]   = [np.mean(values)]
        data[f"MEDIAN_{prefix}"] = [np.median(values)]
        data[f"STD_{prefix}"]    = [np.std(values)]
        
        
    data['operating_hours_estimation_from_means'] = [(data['MEAN_closing_quarter_estimation'][0] - data['MEAN_opening_quarter_estimation'][0]) / 4]
    data['operating_hours_estimation_from_medians'] = [(data['MEDIAN_closing_quarter_estimation'][0] - data['MEDIAN_opening_quarter_estimation'][0]) / 4]
        
    # --------------------------------------------------------------------------------------

    # Return a one-line dataframe
    return pd.DataFrame(data)



# ====================================================================================================
# OPENING AND CLOSING TIMES (PART 2/2)


def get_inflexion_points(curve) :
    """
    Identifies the inflection points in a given curve.

    Parameters:
    ----------
    - curve : array-like
        A sequence of numerical values representing the curve.

    Returns:
    - numpy.ndarray
        Indices of the points in the curve that correspond to inflection points, where the second derivative 
        changes sign (i.e., concavity changes). These points mark shifts in the curvature of the curve.
    """
    
    # Step 1: Compute the first derivative using numpy diff
    first_derivative = np.diff(curve)

    # Step 2: Compute the second derivative
    second_derivative = np.diff(first_derivative)

    # Step 3: Apply a threshold to the second derivative to filter out minor changes
    significant_changes = np.abs(second_derivative) > 0

    # Step 4: Find the points where the second derivative changes sign and passes the threshold
    return np.where((np.diff(np.sign(second_derivative)) != 0) & significant_changes[:-1])[0] + 1  # Adding 1 to correct the index offset


def get_opening_time_feats(df) :
    
    """
    Computes opening time features from a DataFrame containing energy consumption data.
    
    The function performs the following steps:
    1. Computes various statistics (sum, standard deviation, min, max) for energy consumption grouped by time.
    2. Smooths the energy consumption difference using a rolling average.
    3. Interpolates energy consumption data with a polynomial of degree 5 to capture trends and inflection points.
    4. Creates features based on the computed statistics and inflection points.
    
    Parameters:
    - df (pd.DataFrame): A DataFrame containing at least two columns: 'hour_minute' and 'energy'.
                         'hour_minute' should represent the time in HH:MM format, and 'energy' should contain energy consumption values.

    Returns:
    - pd.DataFrame: A DataFrame with the original features and additional computed features related to energy consumption 
                    and its statistics.

    The additional features include:
    - Smoothing adjustments for energy consumption differences.
    - Polynomial coefficients for the fitted polynomial.
    - Number and differences of inflection points in the energy consumption data.
    - Indices of minimum and maximum energy consumption values across time.
    """
    
    # ----------------------------------------------------------------------
    # Part 1 : Compute features based on a groupby operation
    
    # Compute features and groupby
    df['diff'] = df['energy'].diff()
    df['abs_diff'] = abs(df['diff'])
    df_groupby = df.groupby(['hour_minute']).agg({'energy' : ['sum', 'std', 'min', 'max'],
                                                  'diff' : ['sum', 'std'],
                                                  'abs_diff' : ['sum'],
                                                  }).reset_index()
    # Flatten columns
    df_groupby.columns = ['_'.join(i).rstrip('_') for i in df_groupby.columns]
    
    # Smoothen the diff curb (rolling average)
    df_groupby['energy_sum_diff_to_smooth'] = df_groupby['energy_sum']-df_groupby['energy_sum'].rolling(4).mean()
    
    # Standardize
    for k in df_groupby.columns[1:] :
        df_groupby[k] = (df_groupby[k] - df_groupby[k].mean()) / df_groupby[k].std()
        
    # ----------------------------------------------------------------------
    # Part 2 : Interpolate previous data with a polynomial of degree 5
    
    # Set index and stack the dataframe to one single row
    df_tmp = df_groupby.copy()
    df_tmp.index = df_tmp['hour_minute']
    df_tmp = df_tmp.drop(columns = ['hour_minute']).stack().to_frame().reset_index() 
    cols_names = df_tmp['hour_minute'] + '_' + df_tmp['level_1']
    df_tmp = df_tmp[[0]].T
    df_tmp.columns = cols_names
    
    # Rename columns (not to have special character ':' in column names)
    df_tmp = df_tmp.rename(columns = {x : x.replace(':', '_') for x in df_tmp if ':' in x})
    
    # Interpolate with polynomial of degree 5
    data = {}
    for col in ['energy_sum', 'energy_std'] :
        
        # Values
        x = np.arange(0, len(df_groupby), 1)
        y = list(df_groupby[col].values)
        
        # Fit a 5rd-degree polynomial
        polynomial_degree = 5
        coefficients = np.polyfit(x, y, polynomial_degree)
        
        # Generate a polynomial from the coefficients
        poly = np.poly1d(coefficients)
        
        # Create the interpolated curb values using the polynomial
        x_new = np.linspace(x.min(), x.max()+5, len(x)) # Add 5 to get inflexion points out of curb
        y_new = poly(x_new)
        
        # Store poly coefficients
        for i, coef in enumerate(coefficients) :
            data[f'{i+1}th_poly_coef_{col}'] = [coef]
        
        # Plot
        #plt.plot(x, y, 'o', label='Original data')
        #plt.plot(x_new, y_new, '-', label='Cubic interpolation')
        #plt.legend()
        #plt.show()
        
        # Get points of inflexion
        inflexion_points = get_inflexion_points(y_new)
                
        # Store informations
        data[f'n_inflexions_points_{col}'] = [len(inflexion_points)]
        if len(inflexion_points) > 1 :
            data[f'inflexion_diff_{col}'] = [np.max(inflexion_points) - np.min(inflexion_points)]
        else :
            data[f'inflexion_diff_{col}'] = [-1]
            
        for i in range(4) :
            if i < len(inflexion_points) :
                 data[f'{i+1}th_inflexion_point_{col}'] = [inflexion_points[i]]
            else :
                data[f'{i+1}th_inflexion_point_{col}'] = [-1]
                
    # Create a df from data
    data = pd.DataFrame(data)
    
    # Create additional features (argmin and argmax)
    for suffix in ['diff_sum', 'diff_std', 'energy_sum', 'energy_std', 'energy_min', 'energy_max']:
        cols = sorted([x for x in df_tmp if x.endswith(suffix) and len(x)==6 + len(suffix)])
        data[f'argmin_hourminute_{suffix}'] = np.argmin(df_tmp[cols].values, axis=1)
        data[f'argmax_hourminute_{suffix}'] = np.argmax(df_tmp[cols].values, axis=1)
        data[f'diff_argmin_argmax_{suffix}'] = data[f'argmax_hourminute_{suffix}']-data[f'argmin_hourminute_{suffix}']
        mask = (data[f'diff_argmin_argmax_{suffix}'] < 0)
        data.loc[mask, f'diff_argmin_argmax_{suffix}'] += 96 # Add one day (96 quarters)
                        
    # Concat
    df_tmp = pd.concat([df_tmp, data], axis=1)
        
    # Return
    return df_tmp


# ====================================================================================================
# REGION AND STATE FEATURES

dico_state = { # North
               0 : ['NY', 'PA', 'NJ', 'MA', 'MD', 'CT', 'ME', 'NH', 'RI', 'VT', 'DC'],
    
               # Midwest
               1 : ['IL', 'OH', 'MI', 'IN', 'MO', 'WI', 'MN', 'IA', 'KS', 'NE', 'ND', 'SD'],
    
               # West
               2 : ['CA', 'WA', 'CO', 'AZ', 'OR', 'NV', 'UT', 'NM', 'ID', 'MT', 'WY', 'AK', 'HI'],
    
               # South
               3 : ['TX', 'FL', 'NC', 'GA', 'VA', 'TN', 'AL', 'SC', 'LA', 'OK', 'KY', 'MS', 'AR', 'WV'],
    
             }

def get_region(state, dico_state):
    """
    Returns the region for a given state using the dico_state dictionary.
    
    Parameters:
    - state (str): The state abbreviation (e.g., 'NY', 'TX').
    - dico_state (dict): A dictionary mapping region codes to lists of state abbreviations.
    
    Returns:
    - int: Region code (0 for North, 1 for Midwest, 2 for West, 3 for South or default).
    """
    for region, list_of_states in dico_state.items():
        if state in list_of_states :
            return region
    return 3

state_groups = { # Metropolitan areas where there are likely to be many commercial buildings
                0 : ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'NC', 'MI', 'GA', 'VA', 
                     'NJ', 'MO', 'WA', 'MA', 'CO', 'MD', 'AZ', 'IN', 'TN'],
                # Rural or suburban with smaller cities where the landscape is likely to be more residential
                1: ['AL', 'SC', 'LA', 'WI', 'MN', 'OR', 'OK', 'KY', 'MS', 'CT', 
                    'AR', 'NV', 'IA', 'UT', 'NM', 'KS', 'NE', 'ME', 'ID', 
                    'WV', 'NH', 'MT', 'RI', 'DE', 'WY', 'DC', 'ND', 'AK', 
                    'HI', 'SD', 'VT']
            }

def get_state_group(state, dico_state):
    """
    Returns the group for a given state using the state_groups dictionary.
    
    Parameters:
    - state (str): The state abbreviation (e.g., 'CA', 'TX').
    - dico_state (dict): A dictionary mapping group codes to lists of state abbreviations.
    
    Returns:
    - int: Group code (0 for metropolitan areas, 1 for rural/suburban areas).
    """
    for boolean, list_of_states in dico_state.items():
        if state in list_of_states :
            return boolean
    return 1

# Count Encoder
# count_encoder = {k:i for i, k in enumerate(list(df['in.state'].value_counts().index))}
count_encoder = {'CA': 0,
                 'TX': 1,
                 'FL': 2,
                 'NY': 3,
                 'PA': 4,
                 'IL': 5,
                 'OH': 6,
                 'NC': 7,
                 'MI': 8,
                 'GA': 9,
                 'VA': 10,
                 'NJ': 11,
                 'IN': 12,
                 'MO': 13,
                 'TN': 14,
                 'WA': 15,
                 'MA': 16,
                 'CO': 17,
                 'WI': 18,
                 'MD': 19,
                 'AL': 20,
                 'SC': 21,
                 'LA': 22,
                 'MN': 23,
                 'AZ': 24,
                 'OK': 25,
                 'KY': 26,
                 'OR': 27,
                 'MS': 28,
                 'CT': 29,
                 'AR': 30,
                 'NV': 31,
                 'IA': 32,
                 'UT': 33,
                 'NM': 34,
                 'KS': 35,
                 'NE': 36,
                 'ME': 37,
                 'ID': 38,
                 'WV': 39,
                 'NH': 40,
                 'MT': 41,
                 'RI': 42,
                 'DE': 43,
                 'WY': 44,
                 'DC': 45,
                 'ND': 46,
                 'AK': 47,
                 'HI': 48,
                 'SD': 49,
                 'VT': 50}


# ====================================================================================================
# MIN AND MAX VALUES OVER DAYS


def get_max_of_min_over_days(df):
    """
    Computes the maximum of the minimum daily energy values and the minimum of the maximum daily energy values 
    over different time periods (e.g., week, month, season).

    This function aggregates energy consumption data based on specified time periods, calculates the minimum 
    and maximum energy values for each period, and then derives the maximum of the minimum values and 
    minimum of the maximum values across those periods. The results are returned in a structured DataFrame.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the following columns:
        - 'bldg_id' (str): Identifier for the building.
        - 'date' (datetime): Date of the observation.
        - 'energy' (float): Energy consumption values.
        - Time-related columns : 'dayofweek', 'week', 'month', 'season', and 'is_holiday'.

    Returns:
    - final_df (pd.DataFrame): A DataFrame containing:
        - 'bldg_id': Unique building identifiers.
        - Various columns with computed values, including:
            - Maximum of the minimum energy values for different periods.
            - Minimum of the maximum energy values for different periods.
            - Standard deviations of minimum and maximum energy values.
            - Periods at which the maximum of minimums and minimum of maximums were observed.
    """
    
    
    # Initialize a final df
    final_df = df[['bldg_id']].drop_duplicates().reset_index(drop=True)
    
    # Loop over 3 periodicities
    for i, cols_groupby in enumerate([['dayofweek', 'date'],
                                      ['week', 'date'],
                                      ['month', 'date'],
                                      ['season', 'date'],
                                      ['is_holiday', 'date'],
                                     ]):

        # Groupby day + one period column
        df_tmp = df.groupby(cols_groupby).agg({'energy' : [min, max]}).reset_index()

        # Flatten column names
        df_tmp.columns = ['_'.join(i).rstrip('_') for i in df_tmp.columns.values]
        
        # Compute (only one time) the std
        energy_min_std = df_tmp['energy_min'].std()
        energy_max_std = df_tmp['energy_max'].std()
        
        # Groupby cols_groupby[0] (don't use day)
        col_groupby = cols_groupby[0] 
        df_tmp = df_tmp.groupby(col_groupby).agg({'energy_min' : [max], 'energy_max' : [min]}).reset_index()

        # Flatten column names
        df_tmp.columns = ['_'.join(i).rstrip('_') for i in df_tmp.columns.values]

        # Get row names
        df_tmp.index = df_tmp.apply(lambda row : "_".join([f'{col_groupby}{str(row[col_groupby])}']).replace('.0', ''), axis=1)

        # Rename for clarity
        df_tmp = df_tmp.rename(columns = {'energy_min_max' : 'energy_max_of_mins',
                                          'energy_max_min' : 'energy_min_of_maxs'})

        # Notable values
        max_of_mins_MAX = df_tmp["energy_max_of_mins"].max()
        max_of_mins_MIN = df_tmp["energy_max_of_mins"].min()
        min_of_maxs_MAX = df_tmp["energy_min_of_maxs"].max()
        min_of_maxs_MIN = df_tmp["energy_min_of_maxs"].min()

        # Period where the min/max have been reached
        values = df_tmp[col_groupby].values
        period_max_of_mins_MAX = values[df_tmp["energy_max_of_mins"].values.argmax()]
        period_max_of_mins_MIN = values[df_tmp["energy_max_of_mins"].values.argmin()]
        period_min_of_maxs_MAX = values[df_tmp["energy_min_of_maxs"].values.argmax()]
        period_min_of_maxs_MIN = values[df_tmp["energy_min_of_maxs"].values.argmin()]

        # Drop columns
        df_tmp = df_tmp.drop(columns = col_groupby)

        # Create one single row
        data = defaultdict(list)
        for groupby_info, row in df_tmp.iterrows() :
            for compute_info in df_tmp.columns :
                col_name = f"{groupby_info}_{compute_info}"
                data[col_name].append(row[compute_info])
        df_tmp = pd.DataFrame(data=data)

        # Add columns from previously computed values
        df_tmp[f"Max_of_mins_value_MAX_{col_groupby}"] = max_of_mins_MAX
        df_tmp[f"Max_of_mins_value_MIN_{col_groupby}"] = max_of_mins_MIN
        df_tmp[f"Min_of_maxs_value_MAX_{col_groupby}"] = min_of_maxs_MAX
        df_tmp[f"Min_of_maxs_value_MIN_{col_groupby}"] = min_of_maxs_MIN
        df_tmp[f"Max_of_mins_value_MAX_reached_at_{col_groupby}"] = period_max_of_mins_MAX
        df_tmp[f"Max_of_mins_value_MIN_reached_at_{col_groupby}"] = period_max_of_mins_MIN
        df_tmp[f"Min_of_maxs_value_MAX_reached_at_{col_groupby}"] = period_min_of_maxs_MAX
        df_tmp[f"Min_of_maxs_value_MIN_reached_at_{col_groupby}"] = period_min_of_maxs_MIN
        
        # Add the row to the final df
        final_df = pd.concat([final_df, df_tmp], axis=1)
        
    # Add columns
    final_df['energy_min_std'] = energy_min_std
    final_df['energy_max_std'] = energy_max_std
        
    # Return
    return final_df



# ====================================================================================================
# FEATURES ABOUT AMPLITUDES

def get_amplitude_info(df):
    """
    Computes the amplitude (max - min) of energy consumption for different time periods 
    (e.g., week, month, season).

    Parameters:
    - df (pd.DataFrame): A DataFrame containing at least the following columns:
        - 'bldg_id' (str): Identifier for the building.
        - 'date' (datetime): Date of the observation.
        - 'energy' (float): Energy consumption values.
        - Additional time-related columns such as 'week', 'month', and 'season'.

    Returns:
    - final_df (pd.DataFrame): A DataFrame containing:
        - 'bldg_id': Unique building identifiers.
        - Various columns with computed amplitude metrics, including:
            - Amplitude of energy consumption (Max - Min) for each time period.
            - Proportional amplitude relative to the minimum value.
            - Mean and standard deviation of the proportional amplitude for each period.
            - Periods where the maximum and minimum amplitudes were observed.
    """
    
    # Initialize a final df
    final_df = df[['bldg_id']].drop_duplicates().reset_index(drop=True)
    
    # Loop over 3 periodicities
    for col_groupby in ['week',
                        'month',
                        'season',
                       ]:

        # Week by week, check max-min (amplitude)
        df_tmp = df.groupby(col_groupby).agg({'energy' : [min, max]}).reset_index()

        # Flatten column names
        df_tmp.columns = ['_'.join(i).rstrip('_') for i in df_tmp.columns.values]

        # Get row names
        df_tmp.index = df_tmp.apply(lambda row : "_".join([f'{col_groupby}{str(row[col_groupby])}']).replace('.0', ''), axis=1)

        # Compute amplitude
        df_tmp['MaxMinDiff'] = df_tmp['energy_max'] - df_tmp['energy_min']
        df_tmp['MaxMinDiff_proportion'] = (df_tmp['energy_max'] - df_tmp['energy_min']) / df_tmp['energy_min']

        # Drop columns
        df_tmp = df_tmp.drop(columns = [col_groupby, 'energy_max', 'energy_min'])

        # Create one single row
        data = defaultdict(list)
        for groupby_info, row in df_tmp.iterrows() :
            for compute_info in df_tmp.columns :
                col_name = f"{groupby_info}_{compute_info}"
                data[col_name].append(row[compute_info])
        df_tmp = pd.DataFrame(data=data)

        # Compute mean, standard error
        cols_prop = [x for x in df_tmp if 'proportion' in x]
        mean, std = np.mean(df_tmp[cols_prop].values), np.std(df_tmp[cols_prop].values)
        df_tmp[f"{col_groupby}_MaxMinDiff_proportion_mean"] = mean
        df_tmp[f"{col_groupby}_MaxMinDiff_proportion_std"]  = std

        # Find period where the min/max have been reached
        df_tmp[f"MaxMinDiff_maxvalue_reached_at_{col_groupby}"] = df_tmp[cols_prop].values.argmax()
        df_tmp[f"MaxMinDiff_minvalue_reached_at_{col_groupby}"] = df_tmp[cols_prop].values.argmin()
        
        # Add the row to the final df
        final_df = pd.concat([final_df, df_tmp], axis=1)
        
    # Return
    return final_df



# ====================================================================================================
# PEAKS AND HOLLOWS

def identify_peaks_and_hollows(df) :
    """
    Identifies significant peaks and hollows in energy consumption data.

    This function analyzes energy consumption data to categorize periods of high 
    (peaks) and low (hollows) energy consumption based on thresholds derived from
    monthly energy statistics, excluding weekends. It adds boolean columns to the 
    input DataFrame to indicate significant energy consumption events.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing energy consumption data with at least 
      the following columns:
        - 'energy' (float): The energy consumption values.
        - 'month' (int): The month of the year (1 to 12).
        - 'is_weekend' (int): A binary indicator (0 or 1) for whether the day is a weekend.

    Returns:
    - df (pd.DataFrame): The input DataFrame augmented with new columns indicating:
        - 'is_big_energy_consumption_WINTER': True if the energy consumption is above the winter peak threshold.
        - 'is_big_energy_consumption_SUMMER': True if the energy consumption is above the summer peak threshold.
        - 'is_big_energy_consumption': True if the energy consumption is above the threshold for other months.
        - 'is_low_energy_consumption_WINTER': True if the energy consumption is below the winter hollow threshold.
        - 'is_low_energy_consumption_SUMMER': True if the energy consumption is below the summer hollow threshold.
        - 'is_low_energy_consumption': True if the energy consumption is below the threshold for other months.
    """
    
    ######################
    # PEAKS (max values) #
    ######################
    
    # Max values (do not consider week-end)
    values = df[df['is_weekend'] == 0].groupby(['month'])['energy'].max().values

    # WINTER PEAKS
    thr = np.min(values[[0, 1, 11]])
    df['is_big_energy_consumption_WINTER'] = (df['energy'] >= thr)

    # SUMMER
    thr = np.min(values[[6, 7, 8]])
    df['is_big_energy_consumption_SUMMER'] = (df['energy'] >= thr)

    # OTHER
    thr = np.min(values[[4, 5, 10]])
    df['is_big_energy_consumption'] = (df['energy'] >= thr)

    ########################
    # HOLLOWS (min values) #
    ########################

    # Min values (do not consider week-end)
    values = df[df['is_weekend'] == 0].groupby(['month'])['energy'].min().values

    # WINTER PEAKS
    thr = np.max(values[[0, 1, 11]])
    df['is_low_energy_consumption_WINTER'] = (df['energy'] <= thr)

    # SUMMER
    thr = np.max(values[[6, 7, 8]])
    df['is_low_energy_consumption_SUMMER'] = (df['energy'] <= thr)

    # OTHER
    thr = np.max(values[[4, 5, 10]])
    df['is_low_energy_consumption'] = (df['energy'] <= thr)

    # RETURN
    return df


# ====================================================================================================
# FUNCTION TO PUT ALL FEATURES IN A ONE-ROW DATAFRAME


def get_features_on_one_row(df) :
    """
    Extracts and computes various features from energy consumption data for a single building.

    This function processes a DataFrame containing energy consumption data and computes a wide 
    range of features to summarize the energy consumption characteristics for a single building.
    The result is a single-row DataFrame with numerous computed features.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing energy consumption data for a building, 
      which should include the following columns:
        - 'bldg_id' (int): Unique identifier for the building.
        - 'in.state' (str): The state or condition of the building.
        - 'energy' (float): The energy consumption values.
        - 'month' (int): The month of the year (1 to 12).
        - 'is_weekend' (int): A binary indicator (0 or 1) for whether the day is a weekend.
        - 'hour' (int): The hour of the day (0 to 23).
        - 'week' (int): The week of the year (1 to 52).
        - 'dayofweek' (int): The day of the week (0 for Monday to 6 for Sunday).
        - 'day_period' (str): Time period during the day (e.g., 'morning', 'afternoon').
        - Other columns as needed by the various feature extraction functions called within.

    Returns:
    - final_df (pd.DataFrame): A single-row DataFrame containing various computed features, including:
        - Aggregated statistics for energy consumption.
        - Indicators for significant peaks and hollows.
        - Features derived from shifted values and Fourier transforms.
        - Ratios and other manually-computed features related to energy consumption.
    """
    
    # Force 2018 as year to avoid 2019 01-01-01
    df['year'] = 2018
    
    # Identify peaks and hollows
    df = identify_peaks_and_hollows(df)

    # Initialize a final df
    final_df = df[['bldg_id', 'in.state']].iloc[:1].reset_index(drop=True)
    
    # Get info from shifted features
    info = get_features_from_shifted_energy(df)
    final_df = pd.concat([final_df, info], axis=1)
    
    # Get fft info (fourier)
    info = get_fft_feats(df)
    final_df = pd.concat([final_df, info], axis=1)
    
    # Get ramps speeds
    info = get_ramps_speeds(df)
    final_df = pd.concat([final_df, info], axis=1)
    
    # Get info from value_counts
    info = get_value_counts_features(df)
    final_df = pd.concat([final_df, info], axis=1)
    
    # Get info for opening and closing times (part 1/2)
    info = get_opening_and_closing_times(df)
    final_df = pd.concat([final_df, info], axis=1)
    
    # Get info for opening and closing times (part 2/2)
    info = get_opening_time_feats(df)
    final_df = pd.concat([final_df, info], axis=1)
    
    # Create a boolean flag to indicate whether the energy values are equal to the most common value
    most_seen_energy_values = final_df['most_seen_energy_values'].values[0]
    df['is_energy_equal_to_most_common_value'] = ((df['energy'] - most_seen_energy_values) < 1e7).astype(int)
    
    
    ##################################################################################################################
    # DEFINE different dictionnaries of aggregations
    
    # ----------------------------------------------------------------------------------------------------------------
    # SIMPLE Dictionnary of aggregations
    simple_dict_agg = {'energy' : ['min', 'max', 'mean', 'std'],
                       'normalized_energy' : ['mean', 'sum', 'std'],
                       'is_equal_to_last_or_next_quarter' : ['mean'],
                       'is_almost_equal_to_last_or_next_quarter' : ['mean'],
                       'is_energy_equal_to_most_common_value' : ['mean'],
                       'energy_has_increased_since_last_quarter' : ['mean'],
                       'energy_has_decreased_since_last_quarter' : ['mean'],
                       'n_consecutive_quarters_with_increase' : ['mean', 'max'],
                       'n_consecutive_quarters_with_decrease' : ['mean', 'max'],
                       'diff_from_last_quarter' : ['mean', 'sum', 'min', 'max', 'std'],
                       'abs_diff_from_last_quarter' : ['mean', 'sum', 'max', 'std'],
                       'diff_from_last_hour' : ['mean', 'sum', 'max', 'std'],
                       'diff_from_last_week' : ['mean', 'sum', 'max', 'std'],
                       'normalized_energy_diff_from_last_quarter' : ['mean', 'std'],
                       'normalized_energy_diff_from_last_week' : ['mean', 'std'],
                       'normalized_energy_diff_from_last_month' : ['mean', 'std'],
                       'abs_normalized_energy_diff_from_last_quarter' : ['sum'],
                       'abs_normalized_energy_diff_from_last_week' : ['sum'],
                       'abs_normalized_energy_diff_from_last_month' : ['sum'],
                       'is_big_energy_consumption_WINTER' : ['sum'],
                       'is_big_energy_consumption_SUMMER' : ['sum'],
                       'is_big_energy_consumption' : ['sum'],
                       'is_low_energy_consumption_WINTER' : ['sum'],
                       'is_low_energy_consumption_SUMMER' : ['sum'],
                       'is_low_energy_consumption' : ['sum'],
                       'diff_to_day_mean' : ['std'],
                       'diff_to_dayperiod_mean' : ['std'],
                      }
    cols_peak = [x for x in df if 'is_peak' in x]
    for x in cols_peak : simple_dict_agg[x] = ['sum']
    
   
    # ----------------------------------------------------------------------------------------------------------------
    # COMPLETE Dictionnary of aggregations
    complex_dict_agg = {'energy' : ['min', 'max', 'mean', 'sum', 'std', interquantiles_75_25, interquantiles_90_10],
                        'normalized_energy' : ['mean', 'sum', 'std'],
                        'is_equal_to_last_or_next_quarter' : ['mean'],
                        'is_almost_equal_to_last_or_next_quarter' : ['mean'],
                        'is_energy_equal_to_most_common_value' : ['mean'],
                        'energy_has_increased_since_last_quarter' : ['mean'],
                        'energy_has_decreased_since_last_quarter' : ['mean'],
                        'n_consecutive_quarters_with_increase' : ['mean', 'max'],
                        'n_consecutive_quarters_with_decrease' : ['mean', 'max'],
                        'is_big_energy_consumption_WINTER' : ['sum'],
                        'is_big_energy_consumption_SUMMER' : ['sum'],
                        'is_big_energy_consumption' : ['sum'],
                        'is_low_energy_consumption_WINTER' : ['sum'],
                        'is_low_energy_consumption_SUMMER' : ['sum'],
                        'is_low_energy_consumption' : ['sum'],
                        'diff_to_day_mean' : ['std'],
                        'diff_to_dayperiod_mean' : ['std'],
                        'normalized_energy_diff_from_last_quarter' : ['mean', 'std'],
                        'normalized_energy_diff_from_last_week' : ['mean', 'std'],
                        'normalized_energy_diff_from_last_month' : ['mean', 'std'],
                        'abs_normalized_energy_diff_from_last_quarter' : ['mean', 'sum'],
                        'abs_normalized_energy_diff_from_last_week' : ['mean', 'sum'],
                        'abs_normalized_energy_diff_from_last_month' : ['mean', 'sum'],
                       }
    for x in cols_peak : complex_dict_agg[x] = ['sum']
    
    cols = ['diff_from_last_quarter',
            'diff_from_last_2quarters',
            'diff_from_last_3quarters',
            'diff_from_last_hour',
            'diff_from_last_2hour',
            'diff_from_last_week',
            'diff_from_last_month',
           ]
    for k in cols :
        complex_dict_agg[k] = ['mean', 'sum', 'std', 'min', 'max']
        complex_dict_agg['percent_' + k] = ['mean', 'std', 'min', 'max']
        complex_dict_agg['abs_' + k] = ['mean', 'sum', 'max', 'std']
        complex_dict_agg['abs_percent_' + k] = ['mean']
            
    ##################################################################################################################
    # Stats by month
    for dict_agg, cols_groupby in [ 
                                    # SIMPLE GROUPBY
                                    [simple_dict_agg, ['hour']],
                                    [simple_dict_agg, ['week']],
                                    [simple_dict_agg, ['dayofweek', 'day_period']],
                                    [simple_dict_agg, ['is_holiday']],
                                    [simple_dict_agg, ['is_weekend', 'hour_period']],
                                    [simple_dict_agg, ['is_weekend', 'day_period']],
                                    [simple_dict_agg, ['season']],
                                    [simple_dict_agg, ['season', 'is_weekend', 'hour_period']],
                                    
                                    #[simple_dict_agg, ['dayofweek']],
                                    #[simple_dict_agg, ['hour_period']],
                                    #[simple_dict_agg, ['day_period']],
                                    #[simple_dict_agg, ['dayofweek', 'hour'],
       
                                    # COMPLETE GROUPBY
                                    [complex_dict_agg, ['month', 'is_weekend']],
                                    [complex_dict_agg, ['year']],
                                    [complex_dict_agg, ['is_weekend', 'month', 'day_period']],
                                    [complex_dict_agg, ['season', 'is_weekend']],
                                    [complex_dict_agg, ['season', 'is_weekend', 'day_period']],
                                    
                                ] :
                       
        # Create a temporary df with computed values
        df_tmp = df.groupby(cols_groupby).agg(dict_agg).reset_index()
        
        # Flatten column names
        df_tmp.columns = ['_'.join(i).rstrip('_') for i in df_tmp.columns.values]

        # Get row names
        df_tmp.index = df_tmp.apply(lambda row : "_".join([f'{col_name}{str(row[col_name])}' for col_name in cols_groupby]).replace('.0', ''), axis=1)

        # Tranpose df_tmp to have one row per information
        df_tmp = df_tmp.drop(columns = cols_groupby)#.T
        
        # Create one single row : NEW FAST METHOD
        df_tmp = df_tmp.fillna(-1)
        df_tmp = df_tmp.stack().to_frame().reset_index()
        cols_names = df_tmp['level_0'] + '_' + df_tmp['level_1']
        df_tmp = df_tmp[[0]].T
        df_tmp.columns = cols_names
        
        # Add the row to the final df
        final_df = pd.concat([final_df, df_tmp], axis=1)
        
        # Show how many columns are created at each loop
        #print(cols_groupby, final_df.shape)
        
    # ============================================================================================================
    # Complete final_df with ratios and other manually-computed features
    final_df = complete_final_df(final_df)
        
    # ============================================================================================================
    # Add information over amplitudes
    df_tmp   = get_amplitude_info(df).drop(columns = ['bldg_id'])
    final_df = pd.concat([final_df, df_tmp], axis=1)
    
    # ============================================================================================================
    # Add information over max_of_mins and min_of_maxs
    df_tmp   = get_max_of_min_over_days(df).drop(columns = ['bldg_id'])
    final_df = pd.concat([final_df, df_tmp], axis=1)
            
    # Return
    return final_df



# ====================================================================================================
# COMPLETE FINAL DF WITH ADDITIONAL FEATURES

def complete_final_df(final_df) :
    """
    Computes various ratios and metrics from the provided DataFrame to enhance energy consumption analysis.

    This function calculates multiple ratios and progression metrics for energy consumption data over different
    seasons, weekends, and day periods. It aggregates these calculated metrics into the final DataFrame.

    Parameters:
    - final_df (pd.DataFrame): A DataFrame containing various columns related to energy consumption metrics, 
      including mean values for different periods, weekends, and other energy-related features.

    Returns:
    - final_df (pd.DataFrame): The input DataFrame enriched with additional computed metrics, including:
        - Ratios comparing energy consumption across seasons and weekends.
        - Ratios of energy quantiles and means relative to maximum values.
        - Comparisons of the most common energy values to minimum, maximum, and mean energy values.
        - Progression metrics indicating changes in energy consumption over consecutive months.
    """
    
    # Initiate a dictionnary (faster that directly creating column in final_df)
    data = {}
    
    # Compute ratios over seasons
    for col_name in ['energy',
                     'is_equal_to_last_or_next_quarter',
                     'is_energy_equal_to_most_common_value',
                     'diff_from_last_quarter',
                     'diff_from_last_hour',
                     'diff_from_last_week',
                     'diff_from_last_month',
                    ] :
        # Weekend ratios
        data[f'winter_weekend_ratio_{col_name}']    = final_df[f'season0_is_weekend0_{col_name}_mean'] / final_df[f'season0_is_weekend1_{col_name}_mean']
        data[f'spring_weekend_ratio_{col_name}']    = final_df[f'season1_is_weekend0_{col_name}_mean'] / final_df[f'season1_is_weekend1_{col_name}_mean']
        data[f'prespring_weekend_ratio_{col_name}'] = final_df[f'season2_is_weekend0_{col_name}_mean'] / final_df[f'season2_is_weekend1_{col_name}_mean']
        data[f'summer_weekend_ratio_{col_name}']    = final_df[f'season3_is_weekend0_{col_name}_mean'] / final_df[f'season3_is_weekend1_{col_name}_mean']
        data[f'autumn_weekend_ratio_{col_name}']    = final_df[f'season4_is_weekend0_{col_name}_mean'] / final_df[f'season4_is_weekend1_{col_name}_mean']
        # SUMMER ratios
        data[f'summer_over_winter_ratio_{col_name}']    = final_df[f'season3_is_weekend0_{col_name}_mean'] / final_df[f'season0_is_weekend0_{col_name}_mean']
        data[f'summer_over_spring_ratio_{col_name}']    = final_df[f'season3_is_weekend0_{col_name}_mean'] / final_df[f'season1_is_weekend0_{col_name}_mean']
        data[f'summer_over_prespring_ratio_{col_name}'] = final_df[f'season3_is_weekend0_{col_name}_mean'] / final_df[f'season2_is_weekend0_{col_name}_mean']
        data[f'summer_over_autumn_ratio_{col_name}']    = final_df[f'season3_is_weekend0_{col_name}_mean'] / final_df[f'season4_is_weekend0_{col_name}_mean']
        # WINTER ratios
        data[f'winter_over_spring_ratio_{col_name}']    = final_df[f'season0_is_weekend0_{col_name}_mean'] / final_df[f'season1_is_weekend0_{col_name}_mean']
        data[f'winter_over_prespring_ratio_{col_name}'] = final_df[f'season0_is_weekend0_{col_name}_mean'] / final_df[f'season2_is_weekend0_{col_name}_mean']
        data[f'winter_over_autumn_ratio_{col_name}']    = final_df[f'season0_is_weekend0_{col_name}_mean'] / final_df[f'season4_is_weekend0_{col_name}_mean']
        # SUMMER progressions
        data[f'summer_over_winter_progression_{col_name}']    = (final_df[f'season3_is_weekend0_{col_name}_mean'] - final_df[f'season0_is_weekend0_{col_name}_mean']) / final_df[f'season3_is_weekend0_{col_name}_mean']
        data[f'summer_over_spring_progression_{col_name}']    = (final_df[f'season3_is_weekend0_{col_name}_mean'] - final_df[f'season1_is_weekend0_{col_name}_mean']) / final_df[f'season3_is_weekend0_{col_name}_mean']
        data[f'summer_over_prespring_progression_{col_name}'] = (final_df[f'season3_is_weekend0_{col_name}_mean'] - final_df[f'season2_is_weekend0_{col_name}_mean']) / final_df[f'season3_is_weekend0_{col_name}_mean']
        data[f'summer_over_autumn_progression_{col_name}']    = (final_df[f'season3_is_weekend0_{col_name}_mean'] - final_df[f'season4_is_weekend0_{col_name}_mean']) / final_df[f'season3_is_weekend0_{col_name}_mean']
        # SUMMER ratio over day periods
        for day_period in ['night', 'noon', 'morning', 'afternoon'] :
            data[f'summer_over_winter_ratio_{col_name}_afternoon_{day_period}']    = final_df[f'season3_is_weekend0_day_periodafternoon_{col_name}_mean'] / final_df[f'season0_is_weekend0_day_period{day_period}_{col_name}_mean']
            data[f'summer_over_spring_ratio_{col_name}_afternoon_{day_period}']    = final_df[f'season3_is_weekend0_day_periodafternoon_{col_name}_mean'] / final_df[f'season1_is_weekend0_day_period{day_period}_{col_name}_mean']
            data[f'summer_over_autumn_ratio_{col_name}_afternoon_{day_period}']    = final_df[f'season3_is_weekend0_day_periodafternoon_{col_name}_mean'] / final_df[f'season2_is_weekend0_day_period{day_period}_{col_name}_mean']
            data[f'summer_over_prespring_ratio_{col_name}_afternoon_{day_period}'] = final_df[f'season3_is_weekend0_day_periodafternoon_{col_name}_mean'] / final_df[f'season4_is_weekend0_day_period{day_period}_{col_name}_mean']
        
    # Compute ratios : energy_q90/energy_max and energy_mean/energy_max
    data[f'ratio_energy_q90_max'] = final_df['year2018_energy_interquantiles_90_10'] / final_df['year2018_energy_max']
    data[f'ratio_energy_mean_max'] = final_df['year2018_energy_mean'] / final_df['year2018_energy_max']
    for season in range(0, 5) :
        for day_period in ['night', 'noon', 'morning', 'afternoon'] :
            data[f'ratio_energy_q90_max_season{season}_{day_period}']  = final_df[f'season{season}_is_weekend0_day_period{day_period}_energy_interquantiles_90_10'] / final_df[f'season{season}_is_weekend0_day_period{day_period}_energy_max']
            data[f'ratio_energy_mean_max_season{season}_{day_period}'] = final_df[f'season{season}_is_weekend0_day_period{day_period}_energy_mean'] / final_df[f'season{season}_is_weekend0_day_period{day_period}_energy_max']
            
    # Compute feature about most_seen_energy_values
    data['most_seen_energy_values_compared_to_energy_min']  = final_df['most_seen_energy_values'] / final_df['year2018_energy_min']
    data['most_seen_energy_values_compared_to_energy_max']  = final_df['most_seen_energy_values'] / final_df['year2018_energy_max']
    data['most_seen_energy_values_compared_to_energy_mean'] = final_df['most_seen_energy_values'] / final_df['year2018_energy_mean']
    for season in range(0, 5) :
        for day_period in ['night', 'noon', 'morning', 'afternoon'] :
            data[f'most_seen_energy_values_compared_to_energy_min_season{season}_{day_period}']  = final_df['most_seen_energy_values'] / final_df[f'season{season}_is_weekend0_day_period{day_period}_energy_min']
            data[f'most_seen_energy_values_compared_to_energy_max_season{season}_{day_period}']  = final_df['most_seen_energy_values'] / final_df[f'season{season}_is_weekend0_day_period{day_period}_energy_max']
            data[f'most_seen_energy_values_compared_to_energy_mean_season{season}_{day_period}'] = final_df['most_seen_energy_values'] / final_df[f'season{season}_is_weekend0_day_period{day_period}_energy_mean']
                   
    # Feature about week-end
    for season in range(0, 5) :
        for day_period in ['night', 'noon', 'morning', 'afternoon'] :
            data[f'season{season}_period{day_period}_weekend_ratio'] = final_df[f'season{season}_is_weekend0_day_period{day_period}_energy_mean'] / final_df[f'season{season}_is_weekend1_day_period{day_period}_energy_mean']
    
    # Energy progression over consecutive months
    for month in range(1, 12) :
        data[f'month_energy_mean_ratio_m{month+1}_m{month}'] = final_df[f'month{month+1}_is_weekend0_energy_mean'] / final_df[f'month{month}_is_weekend0_energy_mean']
        data[f'month_energy_mean_progression_m{month+1}_m{month}'] = (final_df[f'month{month+1}_is_weekend0_energy_mean'] - final_df[f'month{month}_is_weekend0_energy_mean']) / final_df[f'month{month}_is_weekend0_energy_mean']

    # Concat final_df and data
    final_df = pd.concat([final_df, pd.DataFrame(data=data)], axis=1)
    
    # Return
    return final_df

# ====================================================================================================
# CREATE A DF FROM A SINGLE BUILDING PARQUET FILE

def create_df_from_filepath(filepath):
    """
    This function processes a single parquet file, extracts relevant energy consumption data,
    generates features, and adds region/state information.

    Parameters:
    filepath : str
        Path to the parquet file to be processed.

    Returns:
    df : pandas.DataFrame
        A DataFrame containing processed data from the parquet file, with added date and shifted features, 
        region/state information, and encoded state data.
    """

    # Open file
    df = pd.read_parquet(filepath, engine='pyarrow')

    # Rename and reset_index
    df = df.rename(columns = {'out.electricity.total.energy_consumption' : 'energy'}).reset_index()

    # Create date features
    df = create_date_features(df)

    # Create shifted features
    df = get_shifted_features(df)

    # Compute feature and create a one-line dataframe
    df = get_features_on_one_row(df)

    # Get region & state features
    df['region']      = df['in.state'].apply(lambda state : get_region(state, dico_state))
    df['state_group'] = df['in.state'].apply(lambda state : get_state_group(state, state_groups))

    # Encode in.state (string -> int)
    df['in.state'] = df['in.state'].map(count_encoder)
    
    # Return
    return df

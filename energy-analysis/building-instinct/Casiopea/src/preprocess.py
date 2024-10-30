'''
This script is responsible for generating the features from the consumption time series.
'''

# Load libraries
import numpy as np
import polars as pl
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import glob
import os
import argparse
import sys
import os

DEBUG = False # activate debug mode for developing

def split(a, n):
    '''
    Function to divide a list into n equal parts.
    '''
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def _preprocess(df, col):
    '''
    Here we do the specific preprocess for each batch
    :param df: dataframe with time series
    :param col: type of preprocessing
    :return: final dataframe with the extracted features
    '''
    # Get temporal variables
    tmp_train = df.with_columns([
        pl.col('timestamp').dt.hour().alias('hour'),
        pl.col('timestamp').dt.minute().alias('minute'),
        pl.col('timestamp').dt.month().alias('month'),
        pl.col('timestamp').dt.weekday().alias('weekday'),
        pl.col('timestamp').dt.ordinal_day().alias('yday'),
    ]).with_columns([
        pl.when(pl.col('weekday') <= 5).then(0).otherwise(1).alias('weekend')
    ])


    # Create main variables for each case of preprocessing
    if col=='energy_diff_abs_1':
        tmp_train = tmp_train.sort(by='timestamp',descending=False).with_columns(
            [pl.col('energy').diff(i).over(['bldg_id','train']).alias(f'energy_diff_abs_{i}') for i in [1]]
        )
    elif col=='energy_diff_abs_2':
        tmp_train = tmp_train.sort(by='timestamp',descending=False).with_columns(
            [pl.col('energy').diff(i).over(['bldg_id','train']).alias(f'energy_diff_abs_{i}') for i in [2]]
        )
    elif col=='energy_div_1':
        tmp_train = tmp_train.sort(by='timestamp',descending=False).with_columns(
            [(pl.col('energy') / (pl.col('energy').shift(1) + 0.001)).over(['bldg_id', 'train']).alias(f'energy_div_{i}') for i in [1]]
        )

    ## BASIC FEATURES
    FEATS_DF1 = tmp_train.with_columns([
        (pl.col(col).mean().over(['train', 'state'])).alias(f'{col}_mean_state'),
        (pl.col('energy').max().over(['train', 'bldg_id'])).alias(f'energy_max')
    ])

    # GET BASIS FEATURE BY ID
    uno = FEATS_DF1.group_by(['bldg_id', 'train']).agg([
        (pl.col(col).mean()).alias(f'{col}_mean'),
        (pl.col(col).median()).alias(f'{col}_median'),
        (pl.col(col).max()).alias(f'{col}_max'),
        (pl.col(col).min()).alias(f'{col}_min'),
        (pl.col(col).std()).alias(f'{col}_std'),
        (pl.col(col).kurtosis()).alias(f'{col}_kurt'),
        (pl.col(col).mean() / pl.col(f'{col}_mean_state').first()).alias(f'{col}_mean_state'),
        (pl.col(col).median() / pl.col(f'{col}_mean_state').first()).alias(f'{col}_median_state'),
        (pl.col(col).max() / pl.col(f'{col}_mean_state').first()).alias(f'{col}_max_state'),
        (pl.col(col).min() / pl.col(f'{col}_mean_state').first()).alias(f'{col}_min_state'),
        (pl.col(col).std() / pl.col(f'{col}_mean_state').first()).alias(f'{col}_std_state'),
        (pl.col(col).kurtosis() / pl.col(f'{col}_mean_state').first()).alias(f'{col}_kurt_state'),
    ])

    # GET BASIS FEATURE BY ID AND WEEKDAY
    dos = FEATS_DF1.group_by(['bldg_id','train','weekday']).agg([
        pl.col(col).mean().alias(f'{col}_mean'),
        pl.col(col).median().alias(f'{col}_median'),
        pl.col(col).max().alias(f'{col}_max'),
        pl.col(col).min().alias(f'{col}_min'),
        pl.col(col).std().alias(f'{col}_std'),
    ]).with_columns([
        (pl.col('weekday').cast(str) + '_wd').alias('_feat_by_weekday')
    ]).pivot(index=["bldg_id","train"], values=[f'{col}_mean',f'{col}_median',f'{col}_max',f'{col}_min',f'{col}_std'], columns = "_feat_by_weekday", aggregate_function="first")

    # GET BASIS FEATURE BY ID AND MONTH
    tres = FEATS_DF1.group_by(['bldg_id','train','month']).agg([
        pl.col(col).mean().alias(f'{col}_mean'),
        pl.col(col).median().alias(f'{col}_median'),
        pl.col(col).max().alias(f'{col}_max'),
        pl.col(col).min().alias(f'{col}_min'),
        pl.col(col).std().alias(f'{col}_std'),
    ]).with_columns([
        (pl.col('month').cast(str) + '_month').alias('_feat_by_month')
    ]).pivot(index=["bldg_id","train"], values=[f'{col}_mean',f'{col}_median',f'{col}_max',f'{col}_min',f'{col}_std'], columns = "_feat_by_month", aggregate_function="first")

    # GET BASIS FEATURE BY ID AND HOUR
    cuatro = FEATS_DF1.group_by(['bldg_id','train','hour']).agg([
        pl.col(col).mean().alias(f'{col}_mean'),
        pl.col(col).median().alias(f'{col}_median'),
        pl.col(col).max().alias(f'{col}_max'),
        pl.col(col).min().alias(f'{col}_min'),
        pl.col(col).std().alias(f'{col}_std'),
    ]).with_columns([
        (pl.col('hour').cast(str) + '_h').alias('_feat_by_hour')
    ]).pivot(index=["bldg_id","train"], values=[f'{col}_mean',f'{col}_median',f'{col}_max',f'{col}_min',f'{col}_std'],
             columns = "_feat_by_hour", aggregate_function="first")

    # GET BASIS FEATURE BY ID AND HOUR-15MINUTE WHEN IS NOT WEEKEND
    cinco = FEATS_DF1.filter(pl.col('weekend') == 0).group_by(
        ['bldg_id', 'train', 'hour', 'minute', 'weekend']).agg([
        (pl.col(col).mean() / pl.col('energy_max').first()).alias(f'{col}_mean'),
        (pl.col(col).median() / pl.col('energy_max').first()).alias(f'{col}_median'),
        (pl.col(col).max() / pl.col('energy_max').first()).alias(f'{col}_max'),
        (pl.col(col).min() / pl.col('energy_max').first()).alias(f'{col}_min'),
        (pl.col(col).std() / pl.col('energy_max').first()).alias(f'{col}_std'),
        (pl.col(col).kurtosis() / pl.col('energy_max').first()).alias(f'{col}_kurt'),
    ]).with_columns([
        (pl.col('hour').cast(str) + 'h_' + pl.col('minute').cast(str) + 'm_' + pl.col('weekend').cast(
            str) + 'w').alias('hmw')
    ]).pivot(index=["bldg_id", "train"],
             values=[f'{col}_mean', f'{col}_median', f'{col}_max', f'{col}_min', f'{col}_std'],
             columns="hmw", aggregate_function="first").fill_null(0)

    # Merge all together
    final = uno.join(dos,on=["bldg_id", "train"],how='left',coalesce=True)
    final = final.join(tres, on=["bldg_id", "train"], how='left',coalesce=True)
    final = final.join(cuatro, on=["bldg_id", "train"], how='left',coalesce=True)
    final = final.join(cinco, on=["bldg_id", "train"], how='left',coalesce=True)

    return final

def preprocess(FILE, FEATURE,SAVE_FOLDER):
    '''
    Function to preprocess the data
    :param FILE: name of the file to preprocess
    :param FEATURE: type of preprocessing
    :param SAVE_FOLDER: directory for saving the data
    '''

    # TAG TRAIN (1),TEST (0) OR EXTERNAL DATA (2)
    # We create this numbers for avoiding mix possible matches with bldg_id columns (between different datasets)
    if FILE.find('test') != -1:
        train_num = 0
    elif FILE.find('_ext') != -1:
        train_num = 2
    else:
        train_num = 1

    # READ
    train = pl.read_parquet(FILE).with_columns([pl.lit(train_num).alias('train')])

    # DEBUG MODE
    if DEBUG:
        train = train.filter(pl.col('bldg_id').is_in(train.select(['bldg_id']).unique()[:5]))

    unique_ids = train.select(['bldg_id']).unique()

    # We work in batches of 7000 for avoiding memory issues
    num_parts = 1 + len(unique_ids) // 7000

    batches = list(split(unique_ids, num_parts))

    FINAL_FEATURES = []
    for ix, _ids in tqdm(enumerate(batches)):
        tmp_train = train.filter(pl.col('bldg_id').is_in(_ids))
        tmp_train = _preprocess(tmp_train,FEATURE)
        FEATS_DF = tmp_train
        if ix == 0:
            ord_cols = FEATS_DF.columns
        FINAL_FEATURES.append(FEATS_DF.select(ord_cols))
    FINAL_FEATURES = pl.concat(FINAL_FEATURES)


    try:
        os.makedirs(f'./{SAVE_FOLDER}/{FEATURE}')
    except OSError:
        pass

    FILE = FILE.split("\\")[-1]
    with open(f'./{SAVE_FOLDER}/{FEATURE}/{FILE.split("/")[-1].split(".parquet")[0]}.pkl', 'wb') as f:
        pickle.dump([FINAL_FEATURES,list(np.setdiff1d(FINAL_FEATURES.columns,['bldg_id','train']))], f)

    return FINAL_FEATURES


parser = argparse.ArgumentParser(description="")
parser.add_argument("-rd", "--read_dir", type=str)
parser.add_argument("-sd", "--save_dir", type=str)
parser_args, _ = parser.parse_known_args(sys.argv)


READ_FOLDER = parser_args.read_dir
SAVE_FOLDER = parser_args.save_dir

print(f'READ_FOLDER {READ_FOLDER}')
print(f'SAVE_FOLDER {SAVE_FOLDER}')

# Create directory if it does not exist
try:
    os.makedirs(f'./{SAVE_FOLDER}')
except OSError:
    pass

# Make preprocessing for all files inside the READ_FOLDER
for FILE in glob.glob(f'./{READ_FOLDER}/*.parquet'):#
    if FILE.find('label') == -1 and FILE.find('submission-sample') == -1:
        print(FILE)
        # Make preprocessing for all types of preprocessing that we will use in the models
        for FEATURE in ['energy_div_1','energy_diff_abs_1','energy_diff_abs_2','energy']:
            path = f'./{SAVE_FOLDER}/{FEATURE}/{FILE.split("/")[-1].split(".parquet")[0]}.pkl'
            if not os.path.isfile(path): # Just make the preprocess if it is not done
                print(FEATURE)
                preprocess(FILE, FEATURE,SAVE_FOLDER)
            else:
                print(f'{FILE} -> Already done')


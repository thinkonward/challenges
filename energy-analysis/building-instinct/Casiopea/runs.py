import os
from tqdm import tqdm
import polars as pl
import glob

################################################
### STEP 0 -- DOWNLOAD ALL DATA FROM ONWARD
################################################

# Download data from onward and save it in ./data/onward

# TRAIN -> ./data/onward/building-instinct-train-data
# TRAIN LABELS ->./data/onward/train_label.parquet
# TEST -> ./data/onward/building-instinct-test-data

def group_series_single_dataframe(read_dir,save_dir,name_file):
    # PUT ALL TIME SERIES IN A SINGLE FILE
    ALL = []
    for fn in tqdm(glob.glob(f'{read_dir}/*.parquet')):
        tmp = pl.read_parquet(fn).rename({
            'out.electricity.total.energy_consumption': 'energy',
            'in.state': 'state'
        }).with_columns([
            pl.col('energy').cast(pl.Float32).alias('energy'),
            pl.col('timestamp').dt.cast_time_unit("ms").dt.replace_time_zone(None).alias('timestamp')#.to_datetime("%Y-%m-%d %H:%M%#z")
        ])
        ALL.append(tmp)
    ALL = pl.concat(ALL)

    try:
        os.makedirs(f'{save_dir}')
    except OSError:
        pass

    ALL.write_parquet(f'{save_dir}/{name_file}.parquet')
    return ALL

save_dir = './data/onward'

# GROUP TRAIN DATA
read_dir = './data/onward/building-instinct-train-data'
name_file = 'train_all'
if not os.path.isfile(save_dir + '/' + name_file + '.parquet'):
    group_series_single_dataframe(read_dir, save_dir, name_file)

# GROUP TEST DATA
read_dir = './data/onward/building-instinct-test-data'
name_file = 'test_all'
if not os.path.isfile(save_dir + '/' + name_file + '.parquet'):
    group_series_single_dataframe(read_dir, save_dir, name_file)

# SAVE LABELS ON THE SAME DIRECTORY
if not os.path.isfile('./data/onward/train_label.parquet'):
    labels = pl.read_parquet('./data/onward/building-instinct-train-label/train_label.parquet')
    labels.write_parquet('./data/onward/train_label.parquet')


################################################
### STEP 1 -- DOWNLOAD ALL EXTERNAL DATA
################################################

## EXTERNAL COMMERCIAL DATA DOWNLOADED
os.system('python ./src/download_external_data.py --t commercial --rel comstock_amy2018_release_1 --y 2024 --upgrade 32 --save_dir_raw data/ext_data_2024 --save_dir_pre data/ext_proc_2024')
os.system('python ./src/download_external_data.py --t commercial --rel comstock_amy2018_release_1 --y 2023 --upgrade 10 --save_dir_raw data/ext_data_2023 --save_dir_pre data/ext_proc_2023')
os.system('python ./src/download_external_data.py --t commercial --rel comstock_amy2018_release_2 --y 2023 --upgrade 18 --save_dir_raw data/ext_data_2023_r2 --save_dir_pre data/ext_proc_2023_r2')

## EXTERNAL RESIDENTIAL DATA DOWNLOADED
os.system('python ./src/download_external_data.py --t residential --rel resstock_amy2018_release_2 --y 2024 --upgrade 16 --save_dir_raw data/ext_data_2024 --save_dir_pre data/ext_proc_2024')
os.system('python ./src/download_external_data.py --t residential --rel resstock_amy2018_release_1 --y 2022 --upgrade 10 --save_dir_raw data/ext_data_2022 --save_dir_pre data/ext_proc_2022')


#########################################################################
### STEP 2 -- PREPROCESS ALL DATA AND EXTRACT FEATURES
#########################################################################

os.system('python ./src/preprocess.py --read_dir data/onward --save_dir data/data_ready_onward')
os.system('python ./src/preprocess.py --read_dir data/ext_proc_2024 --save_dir data/data_ready_2024')
os.system('python ./src/preprocess.py --read_dir data/ext_proc_2023 --save_dir data/data_ready_2023')
os.system('python ./src/preprocess.py --read_dir data/ext_proc_2023_r2 --save_dir data/data_ready_2023_r2')
os.system('python ./src/preprocess.py --read_dir data/ext_proc_2022 --save_dir data/data_ready_2022')


### STEP 3 and 4 -> It would be training with all the variables and then running lofo.py to get the importance pickles that we have in the "imp" folder
# I directly attach the 'imp*' pickles to this repo (so there is no need to run it)

#########################################################################
### STEP 5 -- TRAINING
#########################################################################

# Train binary model (commercial or residential)
os.system('python train_binary.py')

# Train metadata models for commercial
os.system('python train.py --training_type commercial --early_stopping')

# Train metadata models for residential
os.system('python train.py --training_type residential --early_stopping')


#########################################################################
### STEP 6 -- INFERENCE
#########################################################################

TEST_DIRECTORY = './data/onward/building-instinct-test-data'  # Where is the time series you want to predict
SAVE_TEST = './private_onward'  # Where to save the unified file
name_file = 'test_private_onward'  # Name of the final unified file
if not os.path.isfile(SAVE_TEST + '/' + name_file + '.parquet'):
    group_series_single_dataframe(TEST_DIRECTORY, SAVE_TEST, name_file)

os.system('python ./src/preprocess.py --read_dir ./private_onward --save_dir ./private_onward/data_ready')

os.system('python inference.py --read_dir ./private_onward/data_ready')





'''
This script is used to download exernal data from data lake:
https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F
Here we have more data similar to the data provided in the competition that we can use for training
'''


import gc
import polars as pl
import glob
from tqdm import tqdm
import argparse
import sys
import os
import shutil

parser = argparse.ArgumentParser(description="")
parser.add_argument("-t", "--type", type=str) # residential or commercial
parser.add_argument("-rel", "--release", type=str)
parser.add_argument("-y", "--year", type=str)
parser.add_argument("-upg", "--upgrade", type=str)
parser.add_argument("-dir_raw", "--save_dir_raw", type=str)
parser.add_argument("-dir_pre", "--save_dir_pre", type=str)
parser_args, _ = parser.parse_known_args(sys.argv)


TYPE = parser_args.type
YEAR = parser_args.year
RELEASE = parser_args.release
UPGRADE = parser_args.upgrade
SAVE_DIR_RAW = parser_args.save_dir_raw # folder where we will save temporally the raw data downloaded
SAVE_DIR_PRE = parser_args.save_dir_pre  # folder where we will save only the useful information from the raw data

print(f'TYPE {TYPE}')
print(f'YEAR {YEAR}')
print(f'RELEASE {RELEASE}')
print(f'UPGRADE {UPGRADE}')
print(f'SAVE_DIR_RAW {SAVE_DIR_RAW}')
print(f'SAVE_DIR_PRE {SAVE_DIR_PRE}')

# Create directory if ti does not exist
try:
    os.makedirs(f'./{SAVE_DIR_RAW}')
except OSError:
    pass
try:
    os.makedirs(f'./{SAVE_DIR_PRE}')
except OSError:
    pass
try:
    os.makedirs(f'./data/labels')
except OSError:
    pass


if TYPE =='residential':
    TYPE_PREFIX = 'res_'
else:
    TYPE_PREFIX = ''

if not os.path.isfile(f'./data/labels/train_label_ext_{TYPE[:3]}_{YEAR}_{RELEASE}.parquet'):
    # DOWNLOAD AND PROCESS LABELS
    os.system(f'aws s3 cp --no-sign-request s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{YEAR}/{RELEASE}/metadata/upgrade{UPGRADE}.parquet labels.parquet')
    print(os.getcwd())
    train_labels = pl.read_parquet('./data/onward/train_label.parquet')
    cols = [i.split(f'_{TYPE[:3]}')[0] for i in train_labels.columns[1:-1] if i.find(f'_{TYPE[:3]}') != -1]
    mapeo = {}
    for i in train_labels.columns[1:-1]:
        if i.find(f'_{TYPE[:3]}') != -1:
            mapeo.update({i.split(f'_{TYPE[:3]}')[0]: i})
    meta1 = pl.read_parquet(f'labels.parquet', columns=['bldg_id', 'in.state'] + cols).rename(mapeo)
    meta1 = meta1.with_columns([
        pl.lit(TYPE).alias('building_stock_type')
    ])
    meta1.write_parquet(f'./data/labels/train_label_ext_{TYPE[:3]}_{YEAR}_{RELEASE}.parquet')
    os.remove(f'labels.parquet')




# List with all the states
mapeo_state = ['ID',
 'NM',
 'NH',
 'CT',
 'LA',
 'IA',
 'UT',
 'OK',
 'NV',
 'MT',
 'OR',
 'MO',
 'KY',
 'WI',
 'MI',
 'VA',
 'SC',
 'NE',
 'SD',
 'AR',
 'KS',
 'WY',
 'NJ',
 'IN',
 'OH',
 'PA',
 'AZ',
 'IL',
 'NY',
 'VT',
 'TN',
 'TX',
 'ND',
 'MS',
 'WA',
 'NC',
 'CA',
 'DE',
 'GA',
 'MA',
 'WV',
 'ME',
 'AK',
 'DC',
 'AL',
 'CO',
 'FL',
 'RI',
 'MD',
 'MN']
# We will download state by state and make a small preprocessing for avoiding memory issues
for CO  in  mapeo_state:

    path = f'./{SAVE_DIR_PRE}/{TYPE_PREFIX}train_all_ext_{CO}.parquet'
    # Check if we already have this file processed
    if not os.path.isfile(path):
        print(f'Starting the download of state: {CO}')
        #os.system(f'aws s3 cp --no-sign-request --recursive s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/comstock_amy2018_release_1/timeseries_individual_buildings/by_state/upgrade=32/state={CO}/ ext_data/state={CO}/')

        try:
            # You need to install aws library in your machine for running this
            os.system(f'aws s3 cp --no-sign-request --recursive s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{YEAR}/{RELEASE}/timeseries_individual_buildings/by_state/upgrade={UPGRADE}/state={CO}/ {SAVE_DIR_RAW}/state={CO}/')


            # Once the download is completed, we read all the parquets files, keep only 3 columns and concatenate all files in one
            ALL = []
            for fn in tqdm(glob.glob(f'./{SAVE_DIR_RAW}/state={CO}/*.parquet')):
                try:
                    tmp = pl.read_parquet(fn,columns=['timestamp','bldg_id','out.electricity.total.energy_consumption'])
                    ALL.append(tmp)
                except:
                    print(f'Prob with  {fn}')

            ALL = pl.concat(ALL)
            gc.collect()
            ALL = ALL.rename({
                        'out.electricity.total.energy_consumption': 'energy'
                    }).with_columns([
                        pl.col('energy').cast(pl.Float32).alias('energy'),
                        pl.col('timestamp').dt.cast_time_unit("ms").dt.replace_time_zone(None).alias('timestamp')
                    ])

            ALL.write_parquet(f'./{SAVE_DIR_PRE}/{TYPE_PREFIX}train_all_ext_{CO}.parquet')
            gc.collect()

            # Once the useful information is saved we can delete all the downloaded information (for avoiding memory issues)
            # Get directory name
            mydir = f'./{SAVE_DIR_RAW}/state={CO}'
            # Try to remove the tree; if it fails, throw an error using try...except.
            try:
                shutil.rmtree(mydir)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

            # For emptying the trash you will need to install trash-empty program in your computer
            os.system('trash-empty')
        except:

            print(f'There was some problem downloading state {CO}')

    else:
        # If the state for this year and release was already downloaded, we will print this
        print(f'{CO} already done')


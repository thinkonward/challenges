'''
This script is where we do the feature selection process
'''

# Load libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
pd.options.mode.copy_on_write = True
import pickle
import polars as pl
import glob
from sklearn.metrics import f1_score
import argparse
import sys
import os

### PARAMETERS
parser = argparse.ArgumentParser(description="")
parser.add_argument("-tt", "--training_type", type=str)
parser.add_argument("-mn", "--model_name", type=str)
parser_args, _ = parser.parse_known_args(sys.str)

TRAINING_TYPE = parser_args.training_type
MODEL_NAME = parser_args.model_name

TARGETS = [
    'in.tstat_clg_sp_f..f_com',
    'in.tstat_htg_sp_f..f_com',
    'in.weekday_opening_time..hr_com',
    'in.weekday_operating_hours..hr_com',
    'in.comstock_building_type_group_com',
    'in.heating_fuel_com',
    'in.hvac_category_com',
    'in.number_of_stories_com',
    'in.ownership_type_com',
    'in.vintage_com',
    'in.wall_construction_type_com',
    'in.bedrooms_res',
    'in.cooling_setpoint_res',
    'in.heating_setpoint_res',
    'in.geometry_building_type_recs_res',
    'in.geometry_floor_area_res',
    'in.geometry_foundation_type_res',
    'in.geometry_wall_type_res',
    'in.heating_fuel_res',
    'in.income_res',
    'in.roof_material_res',
    'in.tenure_res',
    'in.vacancy_status_res',
    'in.vintage_res'
]
TARGETS = [i for i in TARGETS if i.find('_' + TRAINING_TYPE[:3])!=-1]

### STEP 1. READ THE PROCESSED DATA
TYPE_PREPROCESSINGS = ['energy','energy_div_1','energy_diff_abs_1','energy_diff_abs_2']
def read_onward_data(TRAINING_TYPE):
    '''
    Function for reading only training data from onward
    :param TRAINING_TYPE: commercial or residential
    :return: dataframe with all features
    '''
    ALL_SERIES = []
    FEATS = []

    for type_preprocess in TYPE_PREPROCESSINGS:
        print(f'Reading -> {type_preprocess}')

        ALL_SERIES_BY_PREPROCESS = []
        for ix,fn in enumerate(glob.glob(f'./data/data_ready_onward/{type_preprocess}/*.pkl')):

            if fn.find('test')!=-1:
                train_num = 'test'
            else:
                train_num = fn.split('/')[2] + '_' + TRAINING_TYPE

            with open(fn, 'rb') as handle:
                BASIS_FEATS = pickle.load(handle)

            df_all = BASIS_FEATS[0].with_columns([pl.lit(train_num).alias('train')])
            feats = BASIS_FEATS[1]

            if ix==0:
                ord_cols = df_all.columns
            ALL_SERIES_BY_PREPROCESS.append(df_all[ord_cols])

        ALL_SERIES_BY_PREPROCESS = pl.concat(ALL_SERIES_BY_PREPROCESS)
        ALL_SERIES.append(ALL_SERIES_BY_PREPROCESS)
        FEATS = FEATS + feats

    for ix,i in enumerate(ALL_SERIES):
        if ix==0:
            tmp = i
        else:
            tmp = tmp.join(i,on=['bldg_id','train'],how='left',coalesce=True)

    print(f'Dataset size {tmp.shape}')
    return tmp, FEATS
all_val, feats = read_onward_data(TRAINING_TYPE)

# JOIN STATE AND LABELS
train_state= pl.read_parquet('./data/onward/train_all.parquet').with_columns([pl.lit(f'data_ready_onward_{TRAINING_TYPE}').alias('train')])[['bldg_id','train','state']].unique()
train_labels = pl.read_parquet('./data/onward/train_label.parquet').with_columns([pl.lit(f'data_ready_onward_{TRAINING_TYPE}').alias('train')])
train_labels = train_labels.join(train_state,on=['bldg_id','train'],how='left',coalesce=True)

# Create integer feature for state
mapeo_state = {'ID': 0,
 'NM': 1,
 'NH': 2,
 'CT': 3,
 'LA': 4,
 'IA': 5,
 'UT': 6,
 'OK': 7,
 'NV': 8,
 'MT': 9,
 'OR': 10,
 'MO': 11,
 'KY': 12,
 'WI': 13,
 'MI': 14,
 'VA': 15,
 'SC': 16,
 'NE': 17,
 'SD': 18,
 'AR': 19,
 'KS': 20,
 'WY': 21,
 'NJ': 22,
 'IN': 23,
 'OH': 24,
 'PA': 25,
 'AZ': 26,
 'IL': 27,
 'NY': 28,
 'VT': 29,
 'TN': 30,
 'TX': 31,
 'ND': 32,
 'MS': 33,
 'WA': 34,
 'HI': 35,
 'NC': 36,
 'CA': 37,
 'DE': 38,
 'GA': 39,
 'MA': 40,
 'WV': 41,
 'ME': 42,
 'AK': 43,
 'DC': 44,
 'AL': 45,
 'CO': 46,
 'FL': 47,
 'RI': 48,
 'MD': 49,
 'MN': 50}
train_labels = train_labels.with_columns(
    pl.col('state').map_dict(mapeo_state).alias(f"state_int")
)
all_val = all_val.join(train_labels,on=['bldg_id','train'],how='left',coalesce=True)

# JOIN LONGITUDE AND LATITUDE
long_lat = pl.read_csv('./data/long_lat.csv')
long_lat.columns = ['state','long','lat','name_state']
all_val = all_val.join(long_lat,on=['state'],how='left',coalesce=True)

# READ MODEL
with open(f"{MODEL_NAME}.pkl", 'rb') as handle:
    MODELS = pickle.load(handle)

# STAR LOFO PROCESS
imp_by_var = {} # Here we save the importance of each variable

for targ in MODELS.keys():

    # Calculate initial f1-score
    cv_orig = []
    for fold in [0]:
        df_all_tmp = all_val.filter(pl.col('building_stock_type')==TRAINING_TYPE)
        preds =  MODELS[targ][fold][0][0].predict(df_all_tmp[MODELS[targ][fold][0][0].feature_names_].to_pandas())
        f1_metric = f1_score(
            df_all_tmp[targ].to_pandas().map(MODELS[targ][fold][1]),
            np.round(preds),
            average='macro',
        )
        cv_orig.append(f1_metric)
    cv_orig_num = np.mean(cv_orig)
    print(f'{targ}: {cv_orig_num}')

    # See f1-score evolution for each variable
    df_all_p = all_val.filter(pl.col('building_stock_type')==TRAINING_TYPE).to_pandas()
    importance_dict = {}
    secure_features = []
    for var in MODELS[targ][fold][0][0].feature_names_:
        var_lofo = []
        for fold in [0]:
            tmp = df_all_p.copy()
            tmp[var] = tmp[var].sample(frac=1).values
            tmp_preds =  MODELS[targ][fold][0][0].predict(tmp[MODELS[targ][fold][0][0].feature_names_])
            f1_metric = f1_score(
                tmp[targ].map(MODELS[targ][fold][1]),
                np.round(tmp_preds),
                average='macro',
            )

            var_lofo.append(cv_orig[fold] - f1_metric)

        var_lofo_num = np.mean(var_lofo)
        print(f'{targ}, {var}: {var_lofo_num}')

        importance_dict.update({var: var_lofo_num})
    imp_by_var.update({targ:importance_dict})


# Save feature importance
try:
    os.makedirs(f'./imp')
except OSError:
    pass

with open(f'./imp/imp_lofo_{TRAINING_TYPE}.pickle', 'wb') as handle:
    pickle.dump(imp_by_var, handle, protocol=pickle.HIGHEST_PROTOCOL)
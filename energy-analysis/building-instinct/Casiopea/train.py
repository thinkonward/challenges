'''
This script is responsible for training the models for commercial or residential metadata.
Each step will be explained throughout the script.
'''

### Load libraries
import numpy as np
import polars as pl
import pandas as pd
import numpy as np
import pickle
import glob
import random
import os
from catboost import CatBoostClassifier
import argparse
import sys
from sklearn.metrics import f1_score
import os
from scipy import stats

### SPECIFY PARAMETERS
parser = argparse.ArgumentParser(description="")
parser.add_argument("-tt", "--training_type", type=str)
parser.add_argument("-af", "--all_features", action='store_true', help='Activa la opción')
parser.add_argument("-ad", "--train_all_data", action='store_true', help='Activa la opción')
parser.add_argument("-es", "--early_stopping", action='store_true', help='Activa la opción')
parser_args, _ = parser.parse_known_args(sys.argv)


TRAINING_TYPE = parser_args.training_type
ALL_FEATURES = bool(parser_args.all_features)
TRAIN_ALL_DATA = bool(parser_args.train_all_data)
EARLY_STOPPING = bool(parser_args.early_stopping)

print(f'TRAINING_TYPE {TRAINING_TYPE}')
print(f'ALL_FEATURES {ALL_FEATURES}')
print(f'TRAIN_ALL_DATA {TRAIN_ALL_DATA}')
print(f'EARLY_STOPPING {EARLY_STOPPING}')


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

def read_all_data(TRAINING_TYPE):
    '''
    Function for reading all training data
    :param TRAINING_TYPE: commercial or residential
    :return: dataframe with all features
    '''

    # Depend on the training_type we will read commercial time series features or resiential
    pattern = 'res_' if TRAINING_TYPE == 'residential' else 'train'

    ALL_SERIES = []
    FEATS = []

    # We read all types of preprocessing
    for type_preprocess in TYPE_PREPROCESSINGS:
        print(f'Reading -> {type_preprocess}')

        ALL_SERIES_BY_PREPROCESS = []
        for ix,fn in enumerate(np.unique(glob.glob(f'./data/data_ready*/{type_preprocess}/{pattern}*.pkl') + glob.glob(f'./data/data_ready_onward/{type_preprocess}/*.pkl'))):

            # Assign a tag for avoiding possible duplicates in bldg_id
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

            # Append features
            ALL_SERIES_BY_PREPROCESS.append(df_all[ord_cols])

        ALL_SERIES_BY_PREPROCESS = pl.concat(ALL_SERIES_BY_PREPROCESS)
        ALL_SERIES.append(ALL_SERIES_BY_PREPROCESS)
        FEATS = FEATS + feats

    # Put all types of preprocessing on the same dataframe
    for ix,i in enumerate(ALL_SERIES):
        if ix==0:
            tmp = i
        else:
            tmp = tmp.join(i,on=['bldg_id','train'],how='left',coalesce=True)

    print(f'Dataset size {tmp.shape}')
    # return dataframe with all the features and the names of the features
    return tmp, FEATS


### STEP 1. READ ALL DATA

all_train, feats = read_all_data(TRAINING_TYPE)

### STEP 2. ELIMINATE POSSIBLE DUPLICATES
# Since we are downloading more data, there is a possibility that we have the same series that we downloaded
# from onward on the external data.
# Therefore, we will apply some techniques to eliminate totally identical series.


# If any series have exactly same numbers in these agregations columns, we will delete the repeated.
# Because we want to avoid duplicates in our dataset
MAIN_AGREGATIONS = ['energy_mean','energy_median','energy_max','energy_min','energy_std','energy_kurt']

train_orig = all_train.filter(pl.col('train')==f'data_ready_onward_{TRAINING_TYPE}')
external_data = all_train.filter((pl.col('train')!='test') & (pl.col('train')!=f'data_ready_onward_{TRAINING_TYPE}'))
print(f'Train onward -> {train_orig.shape}')
print(f'External training data -> {external_data.shape}')


l1 = list(train_orig.select( value=pl.concat_list(pl.col(MAIN_AGREGATIONS).round(3).cast(str)).list.join(", "))['value'])
l3 = list(external_data.select( value=pl.concat_list(pl.col(MAIN_AGREGATIONS).round(3).cast(str)).list.join(", "))['value'])
external_data = external_data.with_columns([pl.Series(l3).alias('check')])

external_data_cleaned = external_data.filter(~pl.col('check').is_in(l1)).unique(subset=MAIN_AGREGATIONS)
print(f'External training data cleaned -> {external_data_cleaned.shape}')
external_data_cleaned = external_data_cleaned.drop(['check'])

# Concatenate again
all_train = pl.concat([train_orig,external_data_cleaned])


### VALIDATION STRATEGY
# We will use all the data provided by onward as a validation set and all the external data as training
all_train = all_train.with_columns([
    pl.when(pl.col('train')==f'data_ready_onward_{TRAINING_TYPE}').then(0).otherwise(1).alias('fold')
])


### READ AND MERGE LABELS AND STATE FEATURE

# ONWARD DATA
train_state= pl.read_parquet('./data/onward/train_all.parquet').with_columns([pl.lit(f'data_ready_onward_{TRAINING_TYPE}').alias('train')])[['bldg_id','train','state']].unique()
train_labels = pl.read_parquet('./data/onward/train_label.parquet').with_columns([pl.lit(f'data_ready_onward_{TRAINING_TYPE}').alias('train')])
train_labels = train_labels.join(train_state,on=['bldg_id','train'],how='left',coalesce=True)

# EXTERNAL DATA
res_labels_ext_2022 = pl.read_parquet('./data/labels/train_label_ext_res_2022_resstock_amy2018_release_1.parquet').with_columns([pl.lit('data_ready_2022_residential').alias('train')]).rename({'in.state':'state'})
res_labels_ext_2024 = pl.read_parquet('./data/labels/train_label_ext_res_2024_resstock_amy2018_release_2.parquet').with_columns([pl.lit('data_ready_2024_residential').alias('train')]).rename({'in.state':'state'})

com_labels_ext_2023 = pl.read_parquet('./data/labels/train_label_ext_com_2023_comstock_amy2018_release_1.parquet').with_columns([pl.lit('data_ready_2023_commercial').alias('train')]).rename({'in.state':'state'})
com_labels_ext_2023_r2 = pl.read_parquet('./data/labels/train_label_ext_com_2023_comstock_amy2018_release_2.parquet').with_columns([pl.lit('data_ready_2023_r2_commercial').alias('train')]).rename({'in.state':'state'})
com_labels_ext_2024 = pl.read_parquet('./data/labels/train_label_ext_com_2024_comstock_amy2018_release_1.parquet').with_columns([pl.lit('data_ready_2024_commercial').alias('train')]).rename({'in.state':'state'})

def fill_columns(base_df,new_df):
    for c in base_df.columns:
        if c not in new_df.columns:
            new_df = new_df.with_columns([
                pl.lit(None).cast(str).alias(c)
            ])
        elif c != 'bldg_id' and c != 'train':
            new_df = new_df.with_columns([
                pl.col(c).cast(str).alias(c)
            ])
    return new_df

res_labels_ext_2022 = fill_columns(train_labels,res_labels_ext_2022)
res_labels_ext_2024 = fill_columns(train_labels,res_labels_ext_2024)

com_labels_ext_2023 = fill_columns(train_labels,com_labels_ext_2023)
com_labels_ext_2023_r2 = fill_columns(train_labels,com_labels_ext_2023_r2)
com_labels_ext_2024 = fill_columns(train_labels,com_labels_ext_2024)


train_labels = pl.concat([train_labels,
                          res_labels_ext_2022[train_labels.columns],
                          res_labels_ext_2024[train_labels.columns],
                          com_labels_ext_2023[train_labels.columns],
                          com_labels_ext_2023_r2[train_labels.columns],
                          com_labels_ext_2024[train_labels.columns]])

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
    pl.col('state').replace(mapeo_state).alias(f"state_int")
)
all_train = all_train.join(train_labels,on=['bldg_id','train'],how='left',coalesce=True)


# Read and join longitude and latitude information
long_lat = pl.read_csv('./data/long_lat.csv')
long_lat.columns = ['state','long','lat','name_state']
all_train = all_train.join(long_lat,on=['state'],how='left',coalesce=True)


### START TRAINING TARGET BY TARGET
scores = {}
CV = []
ALL_PREDS = []
MODELS =  {}

# In case we don't use early_stopping we will put these numbers of iterations
BEST_ITER = {'in.tstat_clg_sp_f..f_com': 2442.0, 'in.tstat_htg_sp_f..f_com': 1891.0, 'in.weekday_opening_time..hr_com': 1620.0,
             'in.weekday_operating_hours..hr_com': 1633.0, 'in.comstock_building_type_group_com': 1020.0, 'in.heating_fuel_com': 304.0,
             'in.hvac_category_com': 1136.0, 'in.number_of_stories_com': 758.0, 'in.ownership_type_com': 1842.0, 'in.vintage_com': 1309.0,
             'in.wall_construction_type_com': 1277.0,'in.bedrooms_res': 2614.0, 'in.cooling_setpoint_res': 573.0, 'in.heating_setpoint_res': 2897.0,
             'in.geometry_building_type_recs_res': 2780.0, 'in.geometry_floor_area_res': 3818.0, 'in.geometry_foundation_type_res': 421.0,
             'in.geometry_wall_type_res': 664.0, 'in.heating_fuel_res': 1549.0, 'in.income_res': 4691.0, 'in.roof_material_res': 242.0,
             'in.tenure_res': 104.0, 'in.vacancy_status_res': 376.0, 'in.vintage_res': 1680.0}

ALL_REC_FEATS = []
for targ in TARGETS:

    # Apply feature selection if needed
    if not ALL_FEATURES:
        with open(f'./imp/imp_lofo_{TRAINING_TYPE}.pickle', 'rb') as handle:
            imp = pickle.load(handle)
        feats_df = pd.DataFrame.from_dict(imp[targ], orient='index').reset_index()
        feats_df.columns = ['feat','value']
        feats_df = feats_df.sort_values(by=['value'],ascending=False)
        feats = list(feats_df.loc[feats_df['value']>0,'feat'])

    # Make sure there is not duplicates features
    feats = list(set(feats + ['state_int', 'long', 'lat']))
    print(len(feats))
    MODELS_FOLD  = []
    ITERS = []
    CVS = []
    REC_FEATS = []

    # Since we will validate on the training data provided by onward, there is only one fold
    for FOLD in [0]:

        # Convert string target to numerical target
        pos_vals = all_train[targ].unique()
        ver_t = all_train.filter(pl.col(targ).is_in(pos_vals)).filter(pl.col(targ).is_not_null())#.filter(pl.col(targ).is_not_nan())
        mapeo_dict = {}
        for idx, _name in enumerate(sorted(ver_t[targ].unique())):
            mapeo_dict.update({_name:idx})
        ver1 = all_train.filter(pl.col(targ).is_in(pos_vals)).with_columns(
    pl.col(targ).replace(mapeo_dict).alias(f"{targ}_m")
)

        # Specify categorical features
        cat_features = ['state_int']

        # Filter by training_type for security
        if targ.split('_')[-1]=='res':
            ver1 = ver1.filter((pl.col('building_stock_type') == 'residential') )
        else:
            ver1 = ver1.filter((pl.col('building_stock_type') == 'commercial')  )


        TM_MODELS = []
        for i in [0,1,2]:# Number of models for each target

            ## SPLIT TRAINING AND VALIDATION
            # TRAIN
            if TRAIN_ALL_DATA:
                TT = ver1
            else:
                TT = ver1.filter(pl.col('fold') != FOLD)
            X_train = TT[feats]
            y_train = TT[[f"{targ}_m"]]
            print(X_train.shape)
            # VALIDATION
            X_val = ver1.filter(pl.col(f"{targ}_m").is_in(TT[f"{targ}_m"].unique())).filter(pl.col('fold') == FOLD)[feats]
            y_val = ver1.filter(pl.col(f"{targ}_m").is_in(TT[f"{targ}_m"].unique())).filter(pl.col('fold') == FOLD)[[f"{targ}_m"]]
            print(X_val.shape)


            # INTITIALIZE MODEL
            model = CatBoostClassifier(
                                      **{
                        'loss_function' : 'MultiClass',
                        "eval_metric": 'TotalF1:average=Macro',
                        'l2_leaf_reg': 4.622672186898091,
                         'random_state': FOLD,
                         'learning_rate': 0.17985151634618185,
                         'bagging_temperature': 0.0640532793312351,
                         'random_strength': 9.579336738145056,
                         'iterations': 4695 if EARLY_STOPPING else int(BEST_ITER[targ]),
                         'depth': random.randint(5, 8),
                         'border_count': random.randint(50, 125),
                         'verbose': 10,
                        'task_type': "GPU",
            })
            # Train the model
            model.fit(X_train.to_pandas(), y_train.to_pandas(), eval_set=[(X_val.to_pandas(), y_val.to_pandas())], verbose=100, cat_features=cat_features,
                      early_stopping_rounds=300)

            # Store best iteration
            ITERS.append(model.best_iteration_)
            TM_MODELS.append(model)

        # Save model and dictionary of the target (string to number)
        MODELS_FOLD.append([TM_MODELS,mapeo_dict])


        # Make predictions
        preds = stats.mode([np.array(mod.predict(X_val[mod.feature_names_].to_pandas())[:,0],dtype=np.int32) for mod in TM_MODELS],axis=0)
        preds = preds.mode
        preds = [str(i) for i in preds]



        # Calculate validation
        possible_values = list(np.unique(ver1[[f"{targ}_m"]]))
        f1_metric = f1_score(
                y_val.to_pandas(),
                preds,
                labels=possible_values,
                average='macro',
            )
        print(f'{targ} -> {f1_metric}')
        CV.append(model.get_best_score()['validation']['TotalF1:average=Macro'])
        CVS.append(model.get_best_score()['validation']['TotalF1:average=Macro'])


    MODELS.update({targ:MODELS_FOLD})
    BEST_ITER.update({targ:np.mean(ITERS)})
    scores.update({targ: np.mean(CVS)})

# Save models
try:
    os.makedirs(f'./models')
except OSError:
    pass

with open(f'./models/MODELS_{TRAINING_TYPE}.pkl', 'wb') as f:
    pickle.dump(MODELS, f)

# Print final validation scores
print(scores)
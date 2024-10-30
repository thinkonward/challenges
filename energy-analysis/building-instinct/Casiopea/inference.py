'''
This script is responsible for training the models.
Each step will be explained throughout the script.
'''

# Load libraries
import numpy as np
import polars as pl
import pandas as pd
import numpy as np
import pickle
import glob
import random
from catboost import CatBoostClassifier


### STEP 1. READ THE PROCESSED DATA
TYPE_PREPROCESSINGS = ['energy','energy_div_1','energy_diff_abs_1','energy_diff_abs_2']

def read_all_data():
    '''
    Function for reading all training data
    :param TRAINING_TYPE: commercial or residential
    :return: dataframe with all features
    '''

    ALL_SERIES = []
    FEATS = []

    # We read all types of preprocessing
    for type_preprocess in TYPE_PREPROCESSINGS:
        print(f'Reading -> {type_preprocess}')

        TRAINING_TYPE = 'test'

        ALL_SERIES_BY_PREPROCESS = []
        for ix,fn in enumerate(glob.glob(f'./data/data_ready_onward/{type_preprocess}/test*.pkl')):

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

all_test, feats = read_all_data()

# Create integer feature for state
train_state= pl.read_parquet('./data/onward/test_all.parquet')[['bldg_id','state']].unique()
all_test = all_test.join(train_state,on=['bldg_id'],how='left',coalesce=True)
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
all_test = all_test.with_columns(
    pl.col('state').replace(mapeo_state).alias(f"state_int")
)

# Read and join longitude and latitude information
long_lat = pl.read_csv('./data/long_lat.csv')
long_lat.columns = ['state','long','lat','name_state']
long_lat = long_lat.unique(subset=['state'])
all_test = all_test.join(long_lat,on=['state'],how='left',coalesce=True)


def get_preds(MODEL_NAME):
    '''
     Function to which we give the name of the pickle with the models and it returns the predictions for each blgd_id
    :param MODEL_NAME:
    :return: Predictions
    '''

    # Read the model
    with open(f"{MODEL_NAME}.pkl", 'rb') as handle:
        MODELS = pickle.load(handle)

    PREDS = []
    # Iterate through all targets store in the pickle and generate predictions
    for targ in MODELS.keys():

        # Since for each target it is the ensemble of the prediction of three models, we will use this list to store the three predictions
        tmp_preds_folds = []

        # Dictionaries to convert the model outputs into the final categories
        my_map = MODELS[targ][0][1]
        inv_map = {v: k for k, v in my_map.items()}

        # We make the prediction for each model
        for num_model in range(len(MODELS[targ][0][0])):
            preds = MODELS[targ][0][0][num_model].predict(all_test[MODELS[targ][0][0][num_model].feature_names_].to_pandas())[:,0]

            tmp_preds = all_test[['bldg_id']]
            preds = np.vectorize(inv_map.get)(np.array(preds,dtype=np.int32))
            tmp_preds = tmp_preds.with_columns([
                pl.Series(preds).alias(f'{targ}_{num_model}')
            ])
            tmp_preds_folds.append(tmp_preds)

        # Put the three predictions in the same dataframe
        for idx, _df in enumerate(tmp_preds_folds):
            if idx==0:
                pred_df = _df
            else:
                pred_df = pred_df.join(_df,on='bldg_id',how='left',coalesce=True)

        # Our final prediction will be the mode
        pred_df = pred_df.to_pandas()
        pred_df[targ] = pred_df.iloc[:,1:].mode(axis=1)[0]
        PREDS.append(pred_df[['bldg_id',targ]])


    # Join all the targets predicted
    for idx, _df in enumerate(PREDS):
        if idx == 0:
            pred_df = _df
        else:
            pred_df = pd.merge(pred_df,_df, on='bldg_id', how='left')

    return pred_df


### PREDICTIONS
# BINARY
df_b = get_preds('./models/MODELS_binary')
# COMMERCIAL
df_com= get_preds('./models/MODELS_commercial')
# RESIDENTIAL
df_res= get_preds('./models/MODELS_residential')

# Put all together
df = pd.merge(df_b, df_com, on='bldg_id', how='left')
df = pd.merge(df, df_res, on='bldg_id', how='left')

# Put in the appropriate format for submission
ss = pd.read_parquet('./data/onward/building-instinct-submission-sample.parquet')
df = df.sort_values(by='bldg_id', ascending=True)
df.set_index('bldg_id', inplace=True)
df[ss.columns].to_parquet('submission.parquet')
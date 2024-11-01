import pandas as pd
import numpy as np
import os

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def df_process(data_dir):
    load_filepath_labels = os.path.join(data_dir,'labels', 'train_label.parquet')#path to the train label file
    df = pd.read_parquet(load_filepath_labels, engine='pyarrow')
    df_res = df[df["building_stock_type"]=="residential"]
    df_com = df[df["building_stock_type"]=="commercial"]
    res_cols =[col for col in df.columns if col.endswith("res") ]
    com_cols =[col for col in df.columns if col.endswith("com") ]

    df_res.loc[:,'fold'] = -1
    mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    data_labels = df_res[res_cols].values
    for f, (t_, v_) in enumerate(mskf.split(df_res, data_labels)):
        df_res.iloc[v_, df_res.columns.get_loc("fold")] = f
    df_com.loc[:,'fold'] = -1
    mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    data_labels = df_com[com_cols].values
    for f, (t_, v_) in enumerate(mskf.split(df_com, data_labels)):
        df_com.iloc[v_, df_com.columns.get_loc("fold")] = f
    df = pd.concat([df_res,df_com]).sort_index()
    return df


def df_process_ext(data_dir,col_type="res"):
    load_filepath_labels = os.path.join(data_dir,'labels', f'external_{col_type}.parquet')#path to the train label file
    df = pd.read_parquet(load_filepath_labels, engine='pyarrow')
    cols =[col for col in df.columns if col.endswith(col_type) ]
    

    df.loc[:,'fold'] = -1
    mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    data_labels = df[cols].values
    for f, (t_, v_) in enumerate(mskf.split(df, data_labels)):
        df.iloc[v_, df.columns.get_loc("fold")] = f
    
    return df
def main():
    data_dir = '../data'#path to the data directory
    df = df_process(data_dir)
    save_filepath_labels = os.path.join(data_dir,'labels', 'train_label_fold10v1.parquet')#path to save the train label file
    df.to_parquet(save_filepath_labels, engine='pyarrow')
    
    df_ext_res = df_process_ext(data_dir,col_type="res")
    save_filepath_labels = os.path.join(data_dir,'labels', 'external_res_fold10v1.parquet')#path to save the train label file
    df_ext_res.to_parquet(save_filepath_labels, engine='pyarrow')
    
    df_ext_com = df_process_ext(data_dir,col_type="com")
    save_filepath_labels = os.path.join(data_dir,'labels', 'external_com_fold10v1.parquet')#path to save the train label file
    df_ext_com.to_parquet(save_filepath_labels, engine='pyarrow')
    
if __name__ == "__main__":
    main()
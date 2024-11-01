import os
import sys
sys.path.append(os.getcwd())

def get_line_stability() -> None:

    import warnings
    import polars as pl
    
    from src.utils.import_utils import import_config
    from src.eda.distribution_feature import save_multiple_line_plot

    warnings.filterwarnings(action='ignore', category=UserWarning)
    
    config_dict = import_config()
    
    if not os.path.isdir('eda/distribution'):
        os.makedirs('eda/distribution')
    
    label_dataset: pl.LazyFrame = pl.scan_parquet(
        os.path.join(
            config_dict['PATH_GOLD_PARQUET_DATA'],
            f'train_binary_label.parquet'
        )
    )
    feature_dataset: pl.LazyFrame = pl.scan_parquet(
        os.path.join(
            config_dict['PATH_GOLD_PARQUET_DATA'],
            'train_data.parquet'
        )
    )
    feature_list: list[str] = [
        col 
        for col in feature_dataset.collect_schema().names() 
        if col != 'bldg_id'
    ]
    data = (
        label_dataset
        .join(
            feature_dataset,
            on='bldg_id', how='inner'
        )
        .collect()
    )
    save_multiple_line_plot(
        dataset=data, 
        col_to_eda=feature_list,
        save_path='eda/distribution',
        file_name='feature_vs_binary', 
    )
        
if __name__=='__main__':
    get_line_stability()
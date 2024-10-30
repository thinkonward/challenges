import os
import sys
import glob
sys.path.append(os.getcwd())

if __name__ == '__main__':
    """Concatenate testdataset from organizer for inference"""
    import json
    import polars as pl
    
    from tqdm import tqdm
    from src.utils.dtype import remap_category
    from src.utils.import_utils import import_config
    
    
    config_dict = import_config()
    
    with open(
        os.path.join(
            config_dict['PATH_MAPPER_DATA'], 'mapper_category.json'
        ), 'r'            
    ) as file_dtype:
        
        mapper_dataset = json.load(file_dtype)
        
    dataset_chunk_folder: str = os.path.join(
        config_dict['PATH_ORIGINAL_DATA'],
        config_dict['ORIGINAL_TEST_CHUNK_FOLDER']
    )
    data_hour_list: list[pl.DataFrame] = []
    file_list = glob.glob(
        os.path.join(dataset_chunk_folder, '*.parquet')
    )
    for file_path in tqdm(file_list):
        minute_result = (
            pl.scan_parquet(file_path)
            .with_columns(
                pl.col('timestamp').cast(pl.Datetime),
                pl.col('out.electricity.total.energy_consumption').cast(pl.Float64),
                pl.col('in.state').cast(pl.Utf8),
                pl.col('bldg_id').cast(pl.Int64)
            )
            .with_columns(
                pl.col('timestamp').dt.offset_by('-15m')
            )
            .collect()
        )
        hour_result = (
            minute_result
            .group_by(
                'bldg_id', 'in.state', 
                pl.col('timestamp').dt.truncate('1h')
            )
            .agg(
                pl.col('out.electricity.total.energy_consumption').sum()
            )
        )
        data_hour_list.append(hour_result)
        
    data_hour = pl.concat(data_hour_list)
    
    data_hour = remap_category(
        data=data_hour, mapper_mask_col=mapper_dataset['train_data']
    )
        
    data_hour.write_parquet(
        os.path.join(
            config_dict['PATH_SILVER_PARQUET_DATA'],
            config_dict[f'TEST_FEATURE_HOUR_FILE_NAME']
        )
    )
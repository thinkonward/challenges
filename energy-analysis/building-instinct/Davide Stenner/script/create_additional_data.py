import os
import sys

sys.path.append(os.getcwd())

if __name__ == '__main__':
    """Create additional data with scraped home"""
    import gc
    import argparse
    import warnings
    import polars as pl

    from glob import glob
    from tqdm import tqdm
    from typing import Dict
    from src.utils.import_utils import import_config
    from src.utils.logging_utils import get_logger
    from src.utils.dtype import remap_category

    warnings.simplefilter("ignore", pl.exceptions.CategoricalRemappingWarning)
    logger = get_logger(file_name='add_new_data.log')

    config_dict = import_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', default=10_000)
    
    args = parser.parse_args()

    N_SAMPLE: int = args.sample
    
    #import mapper
    mapper_label = import_config(
        os.path.join(
            config_dict['PATH_MAPPER_DATA'], 
            'mapper_category.json'
        )
    )
    folder_map = config_dict['ADDITIONAL_DICT_INFO']

    logger.info('Scanning metadata')

    #COLLECT BUILD ID
    #used only to filter metadata dataset
    bldg_df_dict = {
        'residential': (
            pl.concat(
                [
                    (
                        pl.scan_parquet(path_file)
                        .select('bldg_id')
                        .unique()
                        .sort('bldg_id')
                        .head(N_SAMPLE)
                        .with_columns(pl.lit('residential').alias('build_type').cast(pl.Utf8))
                        .collect()
                    )
                    for path_file in tqdm(
                        glob(
                            f'data_dump/residential/*/*.parquet'
                        )
                    )
                ]
            )
        ),
        'commercial': (
            pl.concat(
                [
                    (
                        pl.scan_parquet(path_file)
                        .select('bldg_id')
                        .unique()
                        .sort('bldg_id')
                        .head(N_SAMPLE)
                        .with_columns(pl.lit('commercial').alias('build_type').cast(pl.Utf8))
                        .collect()
                    )
                    for path_file in tqdm(
                        glob(
                            f'data_dump/commercial/*/*.parquet'
                        )
                    )
                ]
            )
        )
    }
    bldg_df_dict['residential_id'] = bldg_df_dict['residential'].select('bldg_id').unique()
    bldg_df_dict['commercial_id'] = bldg_df_dict['commercial'].select('bldg_id').unique()

    num_residential = bldg_df_dict['residential_id'].select(pl.len()).item()
    num_commercial = bldg_df_dict['commercial_id'].select(pl.len()).item()
    
    logger.info(f'Get {num_residential} residential and {num_commercial} commercial')

    bldg_df = pl.concat(
        [bldg_df_dict['commercial'], bldg_df_dict['residential']]
    )
    #METADATA
    metadata_res = (
        pl.scan_parquet(
            os.path.join(
                'data_dump',
                folder_map['residential']['metadata']
            )
        )
        .select(
            pl.col('bldg_id'),
            pl.lit('residential').alias('build_type'),
            pl.col('in.bedrooms').alias('in.bedrooms_res'),
            pl.col('in.cooling_setpoint').alias('in.cooling_setpoint_res'),
            pl.col('in.heating_setpoint').alias('in.heating_setpoint_res'),
            pl.col('in.geometry_building_type_recs').alias('in.geometry_building_type_recs_res'),
            pl.col('in.geometry_floor_area').alias('in.geometry_floor_area_res'),
            pl.col('in.geometry_foundation_type').alias('in.geometry_foundation_type_res'),
            pl.col('in.geometry_wall_type').alias('in.geometry_wall_type_res'),
            pl.col('in.heating_fuel').alias('in.heating_fuel_res'),
            pl.col('in.income').alias('in.income_res'),
            pl.col('in.roof_material').alias('in.roof_material_res'),
            pl.col('in.tenure').alias('in.tenure_res'),
            pl.col('in.vacancy_status').alias('in.vacancy_status_res'),
            pl.col('in.vintage').alias('in.vintage_res')
        )
    )
    metadata_com = (
        pl.scan_parquet(
            os.path.join(
                'data_dump',
                folder_map['commercial']['metadata']
            )
        )
        .select(
            pl.col('bldg_id'),
            pl.lit('commercial').alias('build_type'),
            pl.col('in.comstock_building_type_group').alias('in.comstock_building_type_group_com'),
            pl.col('in.heating_fuel').alias('in.heating_fuel_com'),
            pl.col('in.hvac_category').alias('in.hvac_category_com'),
            pl.col('in.number_of_stories').alias('in.number_of_stories_com'),
            pl.col('in.ownership_type').alias('in.ownership_type_com'),
            pl.col('in.vintage').alias('in.vintage_com'),
            pl.col('in.wall_construction_type').alias('in.wall_construction_type_com'),
            pl.col('in.tstat_clg_sp_f..f').alias('in.tstat_clg_sp_f..f_com'),
            pl.col('in.tstat_htg_sp_f..f').alias('in.tstat_htg_sp_f..f_com'),
            pl.col('in.weekday_opening_time..hr').alias('in.weekday_opening_time..hr_com'),
            pl.col('in.weekday_operating_hours..hr').alias('in.weekday_operating_hours..hr_com'),
        )
    )

    metadata = (
        pl.concat(
            [metadata_res, metadata_com],
            how='diagonal',
        )
        .collect()
        .join(
            bldg_df.unique(), 
            on=['bldg_id', 'build_type'],
            how='inner'
        )
    )

    logger.info(f'Filtered metadata with {metadata.select(pl.len()).item()}')

    #remap id
    id_build_list: list[str] = (
        metadata
        .select(
            (pl.col('build_type') + pl.col('bldg_id').cast(pl.Utf8)).alias('order_')
        )
        .unique()
        .sort('order_')
        .to_numpy()
        .reshape((-1))
        .tolist()
    )
    mapper_id = {
        key_: 30_000 + id_
        for id_, key_ in enumerate(id_build_list)
    }

    #remap metadata
    metadata = (
        metadata
        .with_columns(
            (
                (
                    pl.col('build_type') + pl.col('bldg_id').cast(pl.Utf8)
                )
                .replace(mapper_id)
                .cast(pl.Int64)
                .alias('bldg_id')
            )
        )
        .rename({'build_type': 'building_stock_type'})
    )

    logger.info('Remapping metadata')
    metadata = remap_category(
        data=metadata, mapper_mask_col=mapper_label['train_label']
    )

    logger.info(f'Starting saving metadata dataset')
    metadata.write_parquet(
        os.path.join(
            config_dict['PATH_SILVER_PARQUET_DATA'],
            'train_label_additional.parquet'
        )
    )

    #import and save label file
    logger.info('importing new data file')
    
    for type_building, type_dict in folder_map.items():
        
        data_hour_list: list[pl.DataFrame] = []
    
        total_number_state = len(
            glob(
                f'data_dump/{type_building}/state=*'
            )
        )
        bar_file = tqdm(total = total_number_state+1)

        type_building_path: str = os.path.join(
            'data_dump', type_dict['path']
        )
        
        state_folder_list = os.listdir(type_building_path)
        
        for state_folder in state_folder_list:
            _ = gc.collect()
            
            state_string: str = state_folder.split('=')[-1]
            
            hour_result = (
                pl.scan_parquet(
                    os.path.join(
                        type_building_path,
                        state_folder,
                        'data.parquet'
                    )
                )
                .filter(
                    pl.col('bldg_id').is_in(bldg_df_dict[type_building + '_id'])
                )
                .select(
                    'timestamp', 'out.electricity.total.energy_consumption', 
                    'in.state', 'bldg_id',
                    pl.lit(type_building).cast(pl.Utf8).alias('build_type')
                )
                .with_columns(
                    (
                        (
                            pl.col('build_type') + pl.col('bldg_id').cast(pl.Utf8)
                        )
                        .replace(mapper_id)
                        .cast(pl.Int64)
                        .alias('bldg_id')
                    )
                )
                .select(
                    ['timestamp', 'out.electricity.total.energy_consumption', 'in.state', 'bldg_id']
                )
                .collect()
            )
                
            state_df: pl.DataFrame = (
                remap_category(
                    data=hour_result, 
                    mapper_mask_col=mapper_label['train_data']
                )
            )

            state_df.write_parquet(
                os.path.join(
                    config_dict['PATH_SILVER_PARQUET_DATA'],
                    f'train_data_{type_building}_additional_{state_string}.parquet'
                )
            )
            del state_df
            _ = gc.collect()
            bar_file.update(1)
            
    
        bar_file.close()
        
        _ = gc.collect()
    
    logger.info(f'Done')
import os
import gc
import json
import glob

import numpy as np
import polars as pl

from tqdm import tqdm
from typing import Any, Tuple, Dict

from src.base.preprocess.pipeline import BasePipeline
from src.preprocess.import_data import PreprocessImport
from src.preprocess.initialize import PreprocessInit
from src.preprocess.add_feature import PreprocessAddFeature
from src.preprocess.cv_fold import PreprocessFoldCreator

class PreprocessPipeline(BasePipeline, PreprocessImport, PreprocessAddFeature, PreprocessFoldCreator):

    def __init__(self, config_dict: dict[str, Any]):
                
        PreprocessInit.__init__(
            self, 
            config_dict=config_dict, 
        )

    def save_data(self) -> None:       
        self.preprocess_logger.info('saving every processed dataset')
        mapper_dummy_target = {}
        
        for name_file, lazy_frame in self.dict_target.items():

            dataset_label = lazy_frame.collect()
                
            self.preprocess_logger.info(f'saving {name_file}')
            (
                dataset_label
                .join(
                    self.data, 
                    on=self.build_id, how='inner'
                )
                .write_parquet(
                os.path.join(
                        self.config_dict['PATH_GOLD_PARQUET_DATA'],
                        f'{name_file}_label.parquet'
                    )
                )
            )

        self.preprocess_logger.info(f'saving target mapper')
        with open(
            os.path.join(
                self.config_dict['PATH_MAPPER_DATA'],
                'target_mapper.json'
            ), 'w'
        ) as file_json:
            json.dump(mapper_dummy_target, file_json)
            
    def collect_feature(self) -> None:
        self.base_data: pl.DataFrame = self.base_data.collect()
        
    def collect_all(self) -> None:
        self.collect_feature()
        
    @property
    def feature_list(self) -> Tuple[str]:
        self.import_all()
        self.create_feature()

        self.merge_all()

        data_columns = self._get_col_name(self.data)
        
        #reset dataset
        self.import_all()
        
        return data_columns
        
    def preprocess_inference(self) -> None:
        self.preprocess_logger.info('Creating feature')
        self.create_feature()

        self.preprocess_logger.info('Merging all')
        self.merge_all()

        self.preprocess_logger.info('Collecting Dataset')

        self.data: pl.DataFrame = self.data.collect()
        self.preprocess_logger.info(
            f'Collected dataset with {len(self._get_col_name(self.data))} columns and {self._get_number_rows(self.data)} rows'
        )

        self.preprocess_logger.info('Saving test dataset')
        self.data.write_parquet(
            os.path.join(
                self.config_dict['PATH_GOLD_PARQUET_DATA'],
                f'test_data.parquet'
            )
        )
        _ = gc.collect()

    def preprocess_train_by_batch(self) -> None:
                
        self.preprocess_logger.info('Safe collecting dataset')
        
        self.preprocess_logger.info('Getting info of all build id')
        build_list = (
            self.label_data
            .select(self.build_id)
            .unique()
            .collect()
            .to_numpy().reshape((-1))
            .tolist()
        )
        data_list: list[pl.DataFrame] = []

        self.economic_data = self.economic_data.collect()
        num_id = len(build_list)
        chunk_size = 10_000
        
        base_data = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_SILVER_PARQUET_DATA'],
                self.config_dict[f'TRAIN_FEATURE_HOUR_FILE_NAME']
            )
        )
        hour_residential = [
            (
                pl.scan_parquet(file_name)
                .select(base_data.collect_schema().names())
            )
            for file_name in glob.glob(
                os.path.join(
                    self.config_dict['PATH_SILVER_PARQUET_DATA'],
                    f'train_data_residential_additional_*.parquet'
                )
            )
        ]
        hour_commercial = [
            (
                pl.scan_parquet(file_name)
                .select(base_data.collect_schema().names())
            )
            for file_name in glob.glob(
                os.path.join(
                    self.config_dict['PATH_SILVER_PARQUET_DATA'],
                    f'train_data_commercial_additional_*.parquet'
                )
            )
        ]

        self.preprocess_logger.info(f'Preprocessing {num_id} building')

        for i in tqdm(range(0, num_id, chunk_size)):
            selected_build_id = build_list[i:min(num_id, i+chunk_size)]
            
            self.base_data: pl.LazyFrame = (
                base_data.clone().filter(pl.col(self.build_id).is_in(selected_build_id))
                .collect()
            )
            
            hour_data_residential = [
                (
                    single_state
                    .clone()
                    .filter(pl.col(self.build_id).is_in(selected_build_id))
                    .collect()
                )
                for single_state in hour_residential
            ]
            hour_data_commercial = [
                (
                    single_state
                    .clone()
                    .filter(pl.col(self.build_id).is_in(selected_build_id))
                    .collect()
                )
                for single_state in hour_commercial
            ]
            
            self.base_data = pl.concat(
                [self.base_data] + 
                hour_data_residential +
                hour_data_commercial
            )
            
            self.downcast_feature()
            
            self.create_feature()
            self.merge_all()

            data_list.append(
                self.data
            )
            self.reset_all_feature_dataset()
            
        self.data = pl.concat(data_list)

        self.preprocess_logger.info(
            f'Collected dataset with {len(self._get_col_name(self.data))} columns and {self._get_number_rows(self.data)} rows'
        )

        _ = gc.collect()
        
        self.preprocess_logger.info('Creating fold_info column ...')
        self.create_fold()
        
        self.preprocess_logger.info('Saving multiple training dataset')
        self.save_data()

    def preprocess_train(self) -> None:
        starting_num_colum = len(self._get_col_name(self.data))
        
        self.preprocess_logger.info('beginning preprocessing training dataset')
        self.preprocess_logger.info('Creating feature')
        self.create_feature()

        self.preprocess_logger.info('Merging all')
        self.merge_all()

        ending_num_colum = len(self._get_col_name(self.data))

        self.preprocess_logger.info(f'Added {ending_num_colum-starting_num_colum} new columns')

        self.preprocess_logger.info('Collecting Dataset')

        self.data: pl.DataFrame = self.data.collect()
        self.preprocess_logger.info(
            f'Collected dataset with {ending_num_colum} columns and {self._get_number_rows(self.data)} rows'
        )

        _ = gc.collect()
        
        self.preprocess_logger.info('Creating fold_info column ...')
        self.create_fold()
        
        self.preprocess_logger.info('Saving multiple training dataset')
        self.save_data()
                
    def begin_training(self) -> None:
        self.import_all()
        
    def begin_inference(self) -> None:
        self.preprocess_logger.info('beginning preprocessing inference dataset')
        
        #reset data
        self.base_data = None
        self.data = None
        self.lazy_feature_list: list[pl.LazyFrame] = []
        
        self.inference: bool = True

    def __call__(self) -> None:
        self.import_all()
        
        if self.inference:    
            self.preprocess_inference()

        else:
            # self.preprocess_train()
            self.preprocess_train_by_batch()
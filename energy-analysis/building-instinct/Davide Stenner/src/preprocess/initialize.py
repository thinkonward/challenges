import os
import json
import logging
import polars as pl
from itertools import product
from typing import Any, Union, Dict

from src.utils.logging_utils import get_logger
from src.base.preprocess.initialize import BaseInit

class PreprocessInit(BaseInit):
    def __init__(self, 
            config_dict: dict[str, Any],
        ):
        self.config_dict: dict[str, Any] = config_dict
        self.n_folds: int = self.config_dict['N_FOLD']

        self.inference: bool = False
        self.additional_data: bool = (
            False if 'add' not in config_dict.keys() else
            config_dict['add']
        )
        
        self._initialize_all()
        self._initialize_preprocess_logger()
        
    def _initialize_all(self) -> None:
        self._initialize_empty_dataset()       
        self._initialize_col_list()
        self._initialize_utils()
        self._initialize_date_list()
    
    def _initialize_date_list(self) -> None:
        self._initialize_dict_mapper()
        
        self.month_list: list[int] = list(range(1, 12+1))
        self.season_list: list[int] = list(range(3))
        self.weekday_list: list[int] = list(range(1, 8))
        self.hour_list: list[int] = list(range(24))
        
        self.tou_unique: list[int] = list(set(self.slice_hour_mapping.values()))
        
    def _initialize_utils(self) -> None:
        self.lazy_feature_list: list[pl.LazyFrame] = []
        
    def _initialize_preprocess_logger(self) -> None:
        self.preprocess_logger: logging.Logger = get_logger('preprocess.txt')
    
    def _initialize_dict_mapper(self) -> None:
        self.month_season_mapping: Dict[int, int] = {
            #cold
            1: 0, 2: 0, 12: 0, 
            #hot
            6: 1, 7: 1, 8: 1,
            #mild
            3: 2, 4: 2, 5: 2,
            9: 2, 10: 2, 11: 2
        }
        #https://co.my.xcelenergy.com/s/billing-payment/residential-rates/time-of-use-pricing
        self.slice_hour_mapping: Dict[int, int] = {
            #off peak
            0: 0, 1: 0, 2:0, 3: 0, 4:0, 5: 0, 6: 0, 7: 0, 
            8: 0, 9: 0, 10: 0, 11: 0, 12: 0,
            20: 0, 21: 0, 22: 0, 23: 0,
            #mid peak
            13: 1, 14: 1, 15: 1,
            #on peak
            16: 2, 17: 2, 18: 2, 19: 2,
        }
            
    def _initialize_col_list(self) -> None:
        self.build_id: str = self.config_dict['BUILDING_ID']
        with open(
            os.path.join(
                self.config_dict['PATH_MAPPER_DATA'], 
                'mapper_category.json'
            ), 'r'            
        ) as file_dtype:
            mapper_ = json.load(file_dtype)
            self.commercial_index = mapper_['train_label']['building_stock_type']['commercial']
            self.state_mapper: Dict[str, int] = mapper_['train_data']['in.state']
            
        self.target_col_binary: str = self.config_dict['TARGET_DICT']['BINARY']
        self.target_col_com_list: list[str] = self.config_dict['TARGET_DICT']['COMMERCIAL']
        self.target_col_res_list: list[str] = self.config_dict['TARGET_DICT']['RESIDENTIAL']
        
        self.all_target_list: list[str] = (
            [self.target_col_binary] +
            self.target_col_com_list +
            self.target_col_res_list
        )
    def _initialize_empty_dataset(self) -> None:
        self.base_data: Union[pl.LazyFrame, pl.DataFrame]
        self.economic_data: Union[pl.LazyFrame, pl.DataFrame]
        self.label_data: Union[pl.LazyFrame, pl.DataFrame]
        self.data: Union[pl.LazyFrame, pl.DataFrame]
        
    def _get_col_name(self, data: Union[pl.DataFrame, pl.LazyFrame]) -> list[str]:
        return data.collect_schema().names()
    
    def _get_number_rows(self, data: Union[pl.DataFrame, pl.LazyFrame]) -> int:
        num_rows = data.select(pl.len())
        
        if isinstance(data, pl.LazyFrame):
            num_rows = num_rows.collect()
            
        return num_rows.item()
    
    def _collect_item_utils(self, data: Union[pl.DataFrame, pl.LazyFrame]) -> Any:
        if isinstance(data, pl.LazyFrame):
            return data.collect().item()
        else:
            return data.item()
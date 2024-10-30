import os
import json
import pickle
import logging

import polars as pl
import lightgbm as lgb

from typing import Any, Union, Dict, Tuple
from src.base.model.initialize import ModelInit
from src.utils.logging_utils import get_logger

class LgbmInit(ModelInit):
    def __init__(self, 
            experiment_name:str, 
            params_lgb: dict[str, Any],
            config_dict: dict[str, Any], data_columns: Tuple[str],
            fold_name: str = 'fold_info'
        ):
        self.inference: bool = False
        self.config_dict: dict[str, Any] = config_dict
        
        self.model_used: list[str] = [
            'building_stock_type', 
            "in.comstock_building_type_group_com", "in.heating_fuel_com",
            "in.hvac_category_com", "in.number_of_stories_com",
            "in.ownership_type_com", "in.vintage_com",
            "in.wall_construction_type_com", "in.tstat_clg_sp_f..f_com",
            "in.tstat_htg_sp_f..f_com", "in.weekday_opening_time..hr_com",
            "in.weekday_operating_hours..hr_com",
            "in.bedrooms_res", "in.cooling_setpoint_res",
            "in.heating_setpoint_res", "in.geometry_building_type_recs_res",
            "in.geometry_floor_area_res", "in.geometry_foundation_type_res",
            "in.geometry_wall_type_res", "in.heating_fuel_res",
            "in.income_res", "in.roof_material_res",
            "in.tenure_res", "in.vacancy_status_res",
            "in.vintage_res"
        ]
        self.binary_model: list[str] = [
            'building_stock_type', "in.tenure_res", "in.vacancy_status_res"
        ]
        self.ordinal_model: list[str] = [
            "in.number_of_stories_com", "in.vintage_com",
            "in.weekday_opening_time..hr_com", "in.weekday_operating_hours..hr_com",
            "in.bedrooms_res", "in.cooling_setpoint_res",
            "in.heating_setpoint_res", "in.geometry_floor_area_res",
            "in.income_res", "in.vintage_res",
            "in.tstat_clg_sp_f..f_com", "in.tstat_htg_sp_f..f_com"
        ]
        self.model_metric_used: list[str] = {
            target_: {'label': 'f1', 'maximize': True}
            for target_ in self.model_used
        }
        self.experiment_path: str = os.path.join(
            config_dict['PATH_EXPERIMENT'],
            experiment_name,
        )
        self.experiment_type_path: str = os.path.join(
            self.experiment_path, '{type}'
        )
        
        self.experiment_path_dict: dict[str, str] = {
            'feature_importance': os.path.join(
                self.experiment_type_path, 'insight'
            ),
            'insight': os.path.join(
                self.experiment_type_path, 'insight'
            ),
            'training': os.path.join(
                self.experiment_type_path, 'training'
            ),
            'shap': os.path.join(
                self.experiment_type_path, 'shap'
            ),
            'model': os.path.join(
                self.experiment_type_path, 'model'
            )
        }

        self.n_fold: int = config_dict['N_FOLD']
        
        self.fold_name: str = fold_name

        self.special_column_list: list[str] = config_dict['SPECIAL_COLUMNS']

        self.useless_col_list: list[str] = (
            self.special_column_list +
            [
                'fold_info', 'current_fold'
            ]
        )

        self.params_lgb: dict[str, Any] = params_lgb
        
        self.feature_list: list[str] = []
        
        self.get_categorical_columns(data_columns=data_columns)
        self.initialize_model_utils()
        self.get_model_file_name_dict()
        self.get_target_col_list()
        
    def get_target_col_list(self) -> None:
        def get_category(x: str) -> str:
            if x[-3:].lower() == 'res':
                return 'residential'
            elif x[-3:].lower() == 'com':
                return 'commercial'
            else:
                return 'binary'
            
        self.target_dict: Dict[str, list[str]] = self.config_dict['TARGET_DICT']
        self.target_class_dict: Dict[str, list[str]] = {
            col_name: get_category(col_name)
            for col_name in self.model_used
        }
    def initialize_logger(self) -> None:
        self.training_logger: logging.Logger = get_logger(
            file_name='training.txt', path_logger=self.experiment_path
        )
        
    def initialize_model_utils(self) -> None:
        self.build_id: str = 'bldg_id'
        
        for type_model in self.model_used:
            setattr(
                self, f'model_{type_model}_list', [] 
            )
            
            setattr(
                self, f'progress_{type_model}_list', [] 
            )

                    
    def get_categorical_columns(self, data_columns: Tuple[str]) -> None:
        #load all possible categorical feature
        cat_col_list = [
            "state"
        ]
        self.categorical_col_list: list[str] = list(
            set(cat_col_list)
            .intersection(set(data_columns))
        )

    def create_experiment_structure(self) -> None:
        if not os.path.isdir(self.experiment_path):
            os.makedirs(self.experiment_path)
            
        for model_type in self.model_used:
            if not os.path.isdir(self.experiment_type_path.format(type=model_type)):
                os.makedirs(self.experiment_type_path.format(type=model_type))
            
            for dir_path_format in self.experiment_path_dict.values():
                dir_path: str = dir_path_format.format(type=model_type)
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)

    def load_model(self, target: str) -> None: 
        self.load_used_feature(target=target)
        self.load_used_categorical_feature(target=target)
        self.load_best_result(target=target)
        self.load_params()
        
        self.load_pickle_model_list(type_model=target)
    
    def get_model_file_name_dict(self) -> None:
        self.model_file_name_dict: dict[str, str] =  {
            'progress_list': {
                type_model: f'progress_{type_model}_list.pkl'
                for type_model in self.model_used
            },
            'best_result': {
                type_model: f'best_result_{type_model}_lgb.txt'
                for type_model in self.model_used
            },
            'model_pickle_list': {
                type_model: f'model_{type_model}_list_lgb.pkl'
                for type_model in self.model_used
            },
            'model_list': {
                type_model: f'lgb_{type_model}' + '_{fold_}.txt'
                for type_model in self.model_used
            }
        }
    
    def save_progress_list(self, progress_list: list, type_model: str) -> None:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=type_model),
                self.model_file_name_dict['progress_list'][type_model]
            ), 'wb'
        ) as file:
            pickle.dump(progress_list, file)

    def load_progress_list(self, type_model: str) -> list:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=type_model),
                self.model_file_name_dict['progress_list'][type_model]
            ), 'rb'
        ) as file:
            progress_list = pickle.load(file)
            
        return progress_list
    
    def save_params(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_lgb.json'
            ), 'w'
        ) as file:
            json.dump(self.params_lgb, file)
    
    def load_params(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_lgb.json'
            ), 'r'
        ) as file:
            self.params_lgb = json.load(file)
    
    def save_best_result(self, best_result: dict[str, Union[int, float]], type_model: str) -> None:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=type_model),
                self.model_file_name_dict['best_result'][type_model]
            ), 'w'
        ) as file:
            json.dump(best_result, file)
        
    def load_best_result(self, type_model: str) -> dict[str, Union[int, float]]:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=type_model),
                self.model_file_name_dict['best_result'][type_model]
            ), 'r'
        ) as file:
            best_result = json.load(file)
            
        return best_result
    
    def save_pickle_model_list(self, model_list: list[lgb.Booster], type_model: str) -> None:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=type_model),
                'model',
                self.model_file_name_dict['model_pickle_list'][type_model]
            ), 'wb'
        ) as file:
            pickle.dump(model_list, file)
    
    def load_pickle_model_list(self, type_model: str) -> list[lgb.Booster]:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=type_model),
                'model',
                self.model_file_name_dict['model_pickle_list'][type_model]
            ), 'rb'
        ) as file:
            model_list = pickle.load(file)
    
        return model_list
    
    def load_model_list(self, type_model: str, file_name: str) -> list[lgb.Booster]:
        
        return [
            lgb.Booster(
                params=self.params_lgb,
                model_file=os.path.join(
                    self.experiment_type_path.format(type=type_model),
                    'model',
                    file_name.format(fold_=fold_)
                )
            )
            for fold_ in range(self.n_fold)
        ]    
            
    def save_used_feature(self, target: str, feature_list: list[str]) -> None:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=target),
                'used_feature.txt'
            ), 'w'
        ) as file:
            json.dump(
                {
                    'feature_model': feature_list
                }, 
                file
            )
    
    def load_used_categorical_feature(self, target: str) -> None:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=target),
                'used_categorical_feature.txt'
            ), 'r'
        ) as file:
            self.categorical_col_list = json.load(file)['categorical_feature']
            
    def save_used_categorical_feature(self, target: str) -> None:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=target),
                'used_categorical_feature.txt'
            ), 'w'
        ) as file:
            json.dump(
                {
                    'categorical_feature': self.categorical_col_list
                }, 
                file
            )

    def load_used_feature(self, target: str) -> None:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=target),
                'used_feature.txt'
            ), 'r'
        ) as file:
            self.feature_list = json.load(file)['feature_model']
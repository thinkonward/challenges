import os
import gc
import numpy as np
import polars as pl
import lightgbm as lgb

from functools import partial
from typing import Tuple, Dict

from src.base.model.training import ModelTrain
from src.model.lgbm.initialize import LgbmInit
from src.model.metric.official_metric import lgb_multi_f1_score, lgb_binary_f1_score, lgb_regression_f1_score

class LgbmTrainer(ModelTrain, LgbmInit):
    def _init_train(self) -> None:
        data = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_GOLD_PARQUET_DATA'],
                'train_building_stock_type_label.parquet'
            )
        )
        
        self.feature_list = [
            col for col in data.collect_schema().names()
            if col not in self.useless_col_list + [self.config_dict['TARGET_DICT']['BINARY']]
        ]
        self.categorical_col_list = [
            col for col in self.categorical_col_list
            if col not in self.useless_col_list
        ]
        self.training_logger.info(f'Using {len(self.categorical_col_list)} categorical features')

    def access_fold(self, fold_: int, current_model: str) -> pl.LazyFrame:
        fold_data = (
            pl.scan_parquet(
                os.path.join(
                    self.config_dict['PATH_GOLD_PARQUET_DATA'],
                    f'train_{current_model}_label.parquet'
                )
            ).with_columns(
                (
                    pl.col('fold_info').str.split(', ')
                    .list.get(fold_).alias('current_fold')
                )
            )
        )
        return fold_data


    def train_target(self, fold_: int, target: str) -> None:
        #classification metric
        params_lgb = self.params_lgb
        if target in self.binary_model:
            params_lgb['objective'] = 'binary'
            params_lgb['num_class'] = 1
            feval = lgb_binary_f1_score
        else:
            params_lgb['objective'] = 'softmax'
            params_lgb['num_class'] = (
                self.access_fold(fold_=fold_, current_model=target)
                .filter(pl.col(target).is_not_null())
                .select(pl.col(target).n_unique())
                .collect()
                .item()
            )
            feval = lgb_multi_f1_score

        progress = {}

        callbacks_list = [
            lgb.record_evaluation(progress),
            lgb.log_evaluation(
                period=50, 
                show_stdv=False
            )
        ]

        train_matrix, test_matrix = self.get_dataset(fold_=fold_, target=target)

        self.training_logger.info(f'Start {target} training')
        model = lgb.train(
            params=params_lgb,
            train_set=train_matrix, 
            num_boost_round=params_lgb['n_round'],
            valid_sets=[test_matrix],
            valid_names=['valid'],
            callbacks=callbacks_list,
            feval=feval
        )

        model.save_model(
            os.path.join(
                self.experiment_type_path.format(type=target),
                (
                    self.model_file_name_dict['model_list'][target]
                    .format(fold_=fold_)
                )
            ), importance_type='gain'
        )

        setattr(
            self, f"model_{target}_list",
            (
                getattr(
                    self, f"model_{target}_list"
                ) +
                [model]
            )
        )
        setattr(
            self, f"progress_{target}_list",
            (
                getattr(
                    self, f"progress_{target}_list"
                ) +
                [progress]
            )
        )

        del train_matrix, test_matrix
        
        _ = gc.collect()

    def get_dataset(self, fold_: int, target: str) -> Tuple[lgb.Dataset]:
        fold_data = self.access_fold(fold_=fold_, current_model=target)
                    
        train_filtered = fold_data.filter(
            (pl.col('current_fold') == 't')
        )
        test_filtered = fold_data.filter(
            (pl.col('current_fold') == 'v')
        )
        
        assert len(
            set(
                train_filtered.select(self.build_id).unique().collect().to_series().to_list()
            ).intersection(
                test_filtered.select(self.build_id).unique().collect().to_series().to_list()
            )
        ) == 0
                        
        train_matrix = lgb.Dataset(
            train_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float64'),
            train_filtered.select(target).collect().to_pandas().to_numpy('float64'),
            feature_name=self.feature_list, categorical_feature=self.categorical_col_list
        )
        test_matrix = lgb.Dataset(
            test_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float64'),
            test_filtered.select(target).collect().to_pandas().to_numpy('float64'),
            feature_name=self.feature_list, categorical_feature=self.categorical_col_list
        )
        return train_matrix, test_matrix

    def train(self) -> None:
        
        self._init_train()
        
        for target_ in self.model_used:
            #save feature list locally for later
            self.save_used_feature(target=target_, feature_list=self.feature_list)
            self.save_used_categorical_feature(target=target_)
            self.training_logger.info(f'Start {target_} with {len(self.feature_list)} features')

            for fold_ in range(self.n_fold):
                self.training_logger.info(f'\n\nStarting fold {fold_}\n\n\n')
                self.training_logger.info('Collecting dataset')
        
                self.train_target(fold_=fold_, target=target_)
            
            self.save_single_model(target=target_)
            
    def save_single_model(self, target: str)->None:            
        self.save_pickle_model_list(
            getattr(
                self, f'model_{target}_list'
            ), 
            target,
        )
        self.save_progress_list(
            getattr(
                self, f'progress_{target}_list'
            ), 
            target
        )
        
    def save_model(self)->None:
        for type_model in self.model_used:
            
            self.save_pickle_model_list(
                getattr(
                    self, f'model_{type_model}_list'
                ), 
                type_model,
            )
            self.save_progress_list(
                getattr(
                    self, f'progress_{type_model}_list'
                ), 
                type_model
            )

        self.save_params()
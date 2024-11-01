import re
import os
import shap
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt

from typing import Union, Tuple, Dict
from itertools import chain
from src.model.lgbm.initialize import LgbmInit
from src.utils.import_utils import import_config

class LgbmExplainer(LgbmInit):       
    def plot_train_curve(self, 
            progress_df: pd.DataFrame, 
            variable_to_plot: Union[str, list],  metric_to_eval: str,
            name_plot: str, type_model: str,
            best_epoch_lgb:int
        ) -> None:
        
        if isinstance(variable_to_plot, str):
            variable_to_plot = [variable_to_plot]
                        
        fig = plt.figure(figsize=(18,8))
        sns.lineplot(
            data=progress_df[['time'] + variable_to_plot].melt(
                id_vars='time',
                value_vars=variable_to_plot,
                var_name='metric_fold', value_name=metric_to_eval
            ), 
            x="time", y=metric_to_eval, hue='metric_fold'
        )
        plt.axvline(x=best_epoch_lgb, color='blue', linestyle='--')

        plt.title(f"Training plot curve of {metric_to_eval}")

        fig.savefig(
            os.path.join(
                self.experiment_path_dict['training'].format(type=type_model),
                f'{name_plot}.png'
            )
        )
        plt.close(fig)

    def evaluate_score(self) -> None:    
        final_score_dict: dict[str, list[float]] = {
            'binary': [],
            'commercial': [],
            'residential': []
        }
        score_dict: Dict[str, float] = {}
        for type_model in self.model_used:
            best_score = self.__evaluate_single_model(type_model=type_model)
            score_dict[type_model] = best_score
            final_score_dict[self.target_class_dict[type_model]].append(best_score)
        
        final_score = (
            final_score_dict['binary'][0] * 0.4 +
            (
                np.mean(final_score_dict['commercial']) * 0.5 +
                np.mean(final_score_dict['residential']) * 0.5
            ) * 0.6
        )
        self.training_logger.info(f'\n\nFinal model pipeline {final_score:.6f}\n\n\n')
        self.__save_final_score(score_dict=score_dict)
        
    def __save_final_score(self, score_dict: Dict[str, float]) -> None:
        base_score = pd.read_excel(
            os.path.join(
                self.config_dict['PATH_OTHER_DATA'],
                'baseline.xlsx'
            )
        )
        base_score['new'] = base_score['feature'].map(score_dict)
        base_score['delta'] = base_score['new'] - base_score['baseline']
        
        agg_base_score = pd.DataFrame(
            {
                'Model': ['baseline', 'new'],
                'COM': [
                    base_score.loc[base_score['Type'] == 'COM', 'baseline'].mean(),
                    base_score.loc[base_score['Type'] == 'COM', 'new'].mean(),                    
                ],
                'RES': [
                    base_score.loc[base_score['Type'] == 'RES', 'baseline'].mean(),
                    base_score.loc[base_score['Type'] == 'RES', 'new'].mean(),                    
                ],
                'TYPE': [
                    base_score.loc[base_score['Type'] == '-', 'baseline'].mean(),
                    base_score.loc[base_score['Type'] == '-', 'new'].mean(),                    
                ]
            }
        )
        agg_base_score['AVG_RES_COM'] = (agg_base_score['COM'] + agg_base_score['RES'])/2
        agg_base_score['TYPE_TOTAL'] = agg_base_score['TYPE'] * 0.4
        agg_base_score['RES_COM_TOTAL'] = agg_base_score['AVG_RES_COM'] * 0.6
        agg_base_score['TOTAL'] = agg_base_score['TYPE_TOTAL'] + agg_base_score['RES_COM_TOTAL']
        
        with pd.ExcelWriter(
            os.path.join(
                self.experiment_path,
                'final_score.xlsx'
            )
        ) as writer:
            
            agg_base_score.to_excel(writer, sheet_name='agg', index=False)
            base_score.to_excel(writer, sheet_name='detail', index=False)

    def __evaluate_single_model(self, type_model: str) -> float:
        metric_eval = self.model_metric_used[type_model]['label']
        metric_to_max = self.model_metric_used[type_model]['maximize']
        
        #load feature list
        self.load_used_feature(target=type_model)
        
        # Find best epoch
        progress_list = self.load_progress_list(
            type_model=type_model
        )

        progress_dict = {}
        list_metric = progress_list[0]['valid'].keys()
        
        for metric_ in list_metric:
            progress_dict.update(
                {
                    f"{metric_}_fold_{i}": progress_list[i]['valid'][metric_]
                    for i in range(self.n_fold)
                }
            )

        progress_df = pd.DataFrame(progress_dict)
        progress_df['time'] = range(progress_df.shape[0])
        
        for metric_ in list_metric:
            
            progress_df[f"average_{metric_}"] = progress_df.loc[
                :, [metric_ in x for x in progress_df.columns]
            ].mean(axis =1)
        
            progress_df[f"std_{metric_}"] = progress_df.loc[
                :, [metric_ in x for x in progress_df.columns]
            ].std(axis =1)

        if metric_to_max:
            best_epoch_lgb = int(progress_df[f"average_{metric_eval}"].argmax())
        else:
            best_epoch_lgb = int(progress_df[f"average_{metric_eval}"].argmin())

        best_score_lgb = progress_df.loc[
            best_epoch_lgb,
            f"average_{metric_eval}"
        ]
        lgb_std = progress_df.loc[
            best_epoch_lgb, f"std_{metric_eval}"
        ]

        self.training_logger.info(f'{type_model} Best epoch: {best_epoch_lgb}, CV-{metric_eval}: {best_score_lgb:.5f} ± {lgb_std:.5f}')

        best_result = {
            'best_epoch': best_epoch_lgb+1,
            'best_score': best_score_lgb
        }
        
        for metric_ in list_metric:
            #plot cv score
            self.plot_train_curve(
                progress_df=progress_df, 
                variable_to_plot=f'average_{metric_}', metric_to_eval=metric_,
                name_plot=f'average_{metric_}_training_curve', type_model=type_model,
                best_epoch_lgb=best_epoch_lgb
            )
            #plot every fold score
            self.plot_train_curve(
                progress_df=progress_df, 
                variable_to_plot=[f'{metric_}_fold_{x}' for x in range(self.n_fold)],
                metric_to_eval=metric_,
                name_plot=f'training_{metric_}_curve_by_fold', type_model=type_model,
                best_epoch_lgb=best_epoch_lgb
            )

        #plot std score
        self.plot_train_curve(
            progress_df=progress_df, 
            variable_to_plot=f'std_{metric_eval}', metric_to_eval=metric_eval,
            name_plot='std_training_curve', type_model=type_model,
            best_epoch_lgb=best_epoch_lgb
        )
        
        self.save_best_result(
            best_result=best_result, type_model=type_model, 
        )
        
        return best_score_lgb
    
    def get_feature_importance(self) -> None:
        result_list_clustered = []
        
        for type_model in self.model_used:
            self.load_used_feature(target=type_model)
            self.__get_single_feature_importance(type_model=type_model)
            result_clustered = self.__get_feature_importance_by_category_feature(current_model=type_model)
            result_clustered['target'] = type_model

            result_list_clustered.append(result_clustered)

        #average value
        result_all_clustered_mean = (
            pd.concat(result_list_clustered, axis=0, ignore_index=True)
            .pivot(
                index=['feature_cluster', 'count'],
                columns='target', values='mean'
            )
            .reset_index()
        )
        result_all_clustered_mean['average_imp'] = result_all_clustered_mean[
            [col for col in result_all_clustered_mean.columns if col not in ['feature_cluster', 'count']]
        ].mean(axis=1)
        
        (
            result_all_clustered_mean
            .to_excel(
                os.path.join(
                    self.experiment_path,
                    'all_feature_importance_clustered.xlsx'
                ), 
                index=False
            )
        )

        #rank
        result_all_clustered = (
            pd.concat(result_list_clustered, axis=0, ignore_index=True)
            .pivot(
                index=['feature_cluster', 'count'],
                columns='target', values='rank'
            )
            .reset_index()
        )
        result_all_clustered['average_rank'] = result_all_clustered[
            [col for col in result_all_clustered.columns if col not in ['feature_cluster', 'count']]
        ].mean(axis=1)
        
        result_all_clustered.to_excel(
            os.path.join(
                self.experiment_path,
                'all_feature_importance_clustered_rank.xlsx'
            ), 
            index=False
        )
    def __get_feature_importance_by_category_feature(self, current_model: str) -> pd.DataFrame:       
        def get_first_if_any(x: list) -> any:
            if len(x)>0:
                return x[0][1:-1]
            else:
                return None
            
        feature_importances = pd.read_excel(
            os.path.join(
                self.experiment_path_dict['feature_importance'].format(type=current_model),
                'feature_importances.xlsx'
            )
        )
        feature_list_cluster: list[int] = [
            r'^' + re.sub(r'\d+', r'\\d+', x) + r'$'
            for x in self.feature_list
        ]
        feature_importances['feature_cluster'] = feature_importances['feature'].apply(
            lambda x:
                get_first_if_any(
                    [
                        pattern_ for pattern_ in feature_list_cluster
                        if bool(re.match(pattern_, x))
                    ]
                )
        )
        feature_importances_cluster = (
            feature_importances
            .groupby('feature_cluster')['average']
            .agg(
                ['mean', 'min', 'max', 'count']
            )
            .sort_values('mean', ascending=False)
            .reset_index()
        )
        self.training_logger.info(
            f"Model {current_model} top2 features are {', '.join(feature_importances_cluster['feature_cluster'].iloc[:2])}\n"
        )
        (
            feature_importances_cluster
            .to_excel(
                os.path.join(
                    self.experiment_path_dict['feature_importance'].format(type=current_model),
                    'feature_importances_clustered.xlsx'
                ), 
                index=False
            )
        )
        feature_importances_cluster = (
            feature_importances_cluster[['feature_cluster', 'count', 'mean']]
            .reset_index(names='rank')
        )
        return feature_importances_cluster
    
    def __get_single_feature_importance(self, type_model: str) -> None:
        best_result = self.load_best_result(
            type_model=type_model
        )
        model_list: list[lgb.Booster] = self.load_pickle_model_list(
            type_model=type_model, 
        )

        feature_importances = pd.DataFrame()
        feature_importances['feature'] = self.feature_list

        for fold_, model in enumerate(model_list):
            feature_importances[f'fold_{fold_}'] = model.feature_importance(
                importance_type='gain', iteration=best_result['best_epoch']
            )

        feature_importances['average'] = feature_importances[
            [f'fold_{fold_}' for fold_ in range(self.n_fold)]
        ].mean(axis=1)
        
        feature_importances = (
            feature_importances[['feature', 'average']]
            .sort_values(by='average', ascending=False)
        )
        self.training_logger.info(
            f"Model {type_model} top2 features are {', '.join(feature_importances['feature'].iloc[:2])}"
        )
        #plain feature
        fig = plt.figure(figsize=(18,8))
        sns.barplot(data=feature_importances.head(50), x='average', y='feature')
        plt.title(f"{type_model} 50 TOP feature importance over {self.n_fold} average")

        fig.savefig(
            os.path.join(
                self.experiment_path_dict['feature_importance'].format(type=type_model), 
                'importance_plot.png'
            )
        )
        plt.close(fig)
        
        #feature importance excel
        feature_importances.to_excel(
            os.path.join(
                self.experiment_path_dict['feature_importance'].format(type=type_model), 
                'feature_importances.xlsx'
            ),
            index=False
        )
    def get_oof_insight(self) -> None:
        pass

    def get_oof_prediction(self) -> None:
        pass
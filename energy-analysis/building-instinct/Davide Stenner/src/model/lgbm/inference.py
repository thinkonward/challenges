import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

from tqdm import tqdm

from src.base.model.inference import ModelPredict
from src.model.lgbm.initialize import LgbmInit

class LgbmInference(ModelPredict, LgbmInit):     
    def load_feature_data(self, data: pl.DataFrame) -> np.ndarray:
        return data.select(self.feature_list).to_pandas().to_numpy(dtype='float64')
        
    def blend_model_predict(self, test_data: pl.DataFrame, model_list: list[lgb.Booster], epoch: int) -> np.ndarray:        
        test_data = self.load_feature_data(test_data)
                
        for iter, model in enumerate(model_list):
            prediction_model = model.predict(
                test_data, num_iteration = epoch
            )/self.n_fold
            
            if iter==0:
                prediction_ = prediction_model
            else:
                prediction_ = np.add(prediction_, prediction_model)
            
        return prediction_
    
    def predict(self, test_data: pl.DataFrame) -> pd.DataFrame:
        assert self.inference

        result_df = pd.DataFrame(
            data=test_data.select(self.build_id).to_numpy(),
            columns=[self.build_id]
        )
        for type_model in tqdm(self.model_used):           
            self.load_used_feature(target=type_model)
            self.load_used_categorical_feature(target=type_model)
            
            best_epoch = self.load_best_result(
                type_model=type_model
            )['best_epoch']
            model_list: list[lgb.Booster] = self.load_pickle_model_list(
                type_model=type_model, 
            )

            prediction_ = self.blend_model_predict(
                test_data=test_data, model_list=model_list, epoch=best_epoch
            )
            if len(prediction_.shape) > 1:
                result_df[type_model] = prediction_.argmax(axis=1)
            else:
                result_df[type_model] = (prediction_>0.5).astype(int)

        return result_df
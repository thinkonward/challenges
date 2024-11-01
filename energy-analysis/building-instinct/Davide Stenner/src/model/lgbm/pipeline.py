from typing import Any, Tuple

from src.model.lgbm.training import LgbmTrainer
from src.model.lgbm.explainer import LgbmExplainer
from src.model.lgbm.initialize import LgbmInit
from src.model.lgbm.inference import LgbmInference
from src.base.model.pipeline import ModelPipeline

class LgbmPipeline(ModelPipeline, LgbmTrainer, LgbmExplainer, LgbmInference):
    def __init__(self, 
            experiment_name:str, 
            params_lgb: dict[str, Any],
            config_dict: dict[str, Any], data_columns: Tuple[str],
            fold_name: str = 'fold_info', 
            evaluate_shap: bool=False
        ):
        LgbmInit.__init__(
            self, experiment_name=experiment_name, params_lgb=params_lgb,
            config_dict=config_dict,
            data_columns=data_columns, 
            fold_name=fold_name
        )
        self.evaluate_shap: bool = evaluate_shap
        
    def activate_inference(self) -> None:
        self.inference = True
        
    def run_train(self) -> None:
        self.save_params()
        self.train()
        
    def explain_model(self) -> None:
        self.evaluate_score()
        self.get_feature_importance()
        self.get_oof_prediction()
        self.get_oof_insight()
        
    def train_explain(self) -> None:
        self.create_experiment_structure()
        self.initialize_logger()
        self.run_train()
        self.explain_model()
from abc import ABC, abstractmethod

class ModelExplainer(ABC):
    @abstractmethod
    def plot_train_curve(self) -> None:
        pass
    
    @abstractmethod
    def evaluate_curve(self) -> None: 
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> None: 
        pass
    
    @abstractmethod
    def get_oof_prediction(self) -> None:
        pass
from abc import ABC, abstractmethod

class ModelPipeline(ABC):
    @abstractmethod
    def activate_inference(self) -> None:
        pass
    
    @abstractmethod
    def run_train(self) -> None: 
        pass
    
    @abstractmethod
    def explain_model(self) -> None:
        pass
    
    @abstractmethod
    def train_explain(self) -> None:
        pass
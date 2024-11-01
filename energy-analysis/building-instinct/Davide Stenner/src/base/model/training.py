from abc import ABC, abstractmethod

class ModelTrain(ABC):
    @abstractmethod
    def _init_train(self) -> None:
        pass
    
    @abstractmethod
    def train(self) -> None: 
        pass
    
    @abstractmethod
    def save_model(self) -> None: 
        pass
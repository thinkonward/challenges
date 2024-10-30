from abc import ABC, abstractmethod

class ModelInit(ABC):
    @abstractmethod
    def create_experiment_structure(self) -> None:
        pass
    
    @abstractmethod
    def load_model(self) -> None: 
        pass
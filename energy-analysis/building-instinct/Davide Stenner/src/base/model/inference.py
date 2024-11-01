from abc import ABC, abstractmethod

class ModelPredict(ABC):    
    @abstractmethod
    def predict(self) -> None: 
        pass
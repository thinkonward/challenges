from abc import ABC, abstractmethod

class BaseInit(ABC):
    
    @abstractmethod
    def _initialize_empty_dataset(self) -> None:
        pass
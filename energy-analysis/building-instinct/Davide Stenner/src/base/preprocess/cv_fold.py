from abc import ABC, abstractmethod

class BaseCVFold(ABC):

    @abstractmethod
    def create_fold(self) -> None:
        pass        

from abc import ABC, abstractmethod

class BaseFeature(ABC):

    @abstractmethod
    def create_feature(self) -> None:
        pass

    @abstractmethod
    def merge_all(self) -> None:
        pass
from abc import ABC, abstractmethod

class BaseImport(ABC):

    @abstractmethod
    def scan_all_dataset(self) -> None:
        pass

    @abstractmethod
    def import_all(self) -> None:
        pass
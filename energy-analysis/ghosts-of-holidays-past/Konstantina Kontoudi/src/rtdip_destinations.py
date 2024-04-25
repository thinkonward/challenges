import pickle
from pathlib import Path
from typing import Union

import pandas as pd
from rtdip_sdk.pipelines._pipeline_utils.models import Libraries, SystemType
from rtdip_sdk.pipelines.destinations.interfaces import DestinationInterface


class DiskDestination(DestinationInterface):
    data: object

    def __init__(self, data: object, path: Union[str, Path], file_type: str, query_name: str = "") -> None:
        # Added query_name because the executor passes it to the DestinationInterface factory
        # Otherwise the pipeline execution fails
        self.data = data
        self.path = path
        self.file_type = file_type

    def pre_write_validation(self) -> bool:
        return True

    def post_write_validation(self) -> bool:
        return True

    def write_batch(self):
        if self.file_type == "parquet":
            if not isinstance(self.data, pd.DataFrame):
                raise NotImplementedError(f"Cannot dump to parquet format object of type {type(self.data)}. Only "
                                          f"pd.DataFrame is supported")
            self.data.to_parquet(self.path)
        elif self.file_type == "pickle":
            with open(self.path, 'wb') as fh:
                pickle.dump(self.data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise NotImplementedError(f"File type {self.file_type} not supported")

    def write_stream(self):
        """
        Raises: NotImplementedError
        """
        raise NotImplementedError(
            "Writing streams to disk using python is not implemented"
        )

    @staticmethod
    def system_type() -> SystemType:
        """
        Attributes:
            SystemType (Environment): Requires PYTHON
        """
        return SystemType.PYTHON

    @staticmethod
    def libraries() -> Libraries:
        libraries = Libraries()
        return libraries

    @staticmethod
    def settings() -> dict:
        return {}

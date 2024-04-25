from typing import Callable, Union, List

from rtdip_sdk.pipelines._pipeline_utils.models import Libraries, SystemType
from rtdip_sdk.pipelines.transformers.interfaces import TransformerInterface


class FunctionTransformer(TransformerInterface):
    data: Union[object, List[object]]

    def __init__(
            self, data: Union[object, List[object]], function: Callable
    ) -> None:
        self.data = data
        self.function = function

    @staticmethod
    def system_type():
        """
        Attributes:
            SystemType (Environment): Requires PYSPARK
        """
        return SystemType.PYTHON

    @staticmethod
    def libraries():
        libraries = Libraries()
        return libraries

    @staticmethod
    def settings() -> dict:
        return {}

    def pre_transform_validation(self):
        return True

    def post_transform_validation(self):
        return True

    def transform(self) -> object:
        """
        Returns:
            object: The object returned by the function passed as an input
        """
        if isinstance(self.data, list):
            return self.function(*self.data)
        else:
            return self.function(self.data)

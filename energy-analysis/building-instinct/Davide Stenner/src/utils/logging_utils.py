import os
import logging

from src.utils.import_utils import import_config

def remove_logger(logger: logging.Logger, file_name: str, path_logger: str = None) -> None:
    if path_logger is None:
        config = import_config()
        path_logger = config['LOG_FOLDER']
        
    file_handler = logging.FileHandler(
        os.path.join(
            path_logger, 
            file_name
        ), mode='w'
    )
    console_handlare = logging.StreamHandler()
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handlare)
    del logger
    
def get_logger(file_name: str, path_logger: str = None) -> logging.Logger:
    
    if path_logger is None:
        config = import_config()
        path_logger = config['LOG_FOLDER']
        
    logger: logging.Logger = logging.getLogger()
    if not logger.handlers:

        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(
            os.path.join(
                path_logger, 
                file_name
            ), mode='w'
        )
        console_handlare = logging.StreamHandler()

        logger.addHandler(file_handler)
        logger.addHandler(console_handlare)

    return logger
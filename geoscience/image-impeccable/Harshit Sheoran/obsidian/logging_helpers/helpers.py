import logging
from tqdm import tqdm
import time
from io import StringIO

class StringCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.stream = StringIO()

    def emit(self, record):
        # Write the formatted log record into the StringIO buffer
        msg = self.format(record)
        self.stream.write(msg)
        self.stream.seek(0)  # Reset cursor to the beginning of the captured message

    def get_log(self):
        self.stream.seek(0)  # Reset cursor
        return self.stream.read().strip()  # Read the message and strip any trailing newlines
    
class TqdmLogger(tqdm):
    def __init__(self, logger, string_handler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.string_handler = string_handler
        self.bar_format = '{desc}'  # Only show the description

    def log(self, message):
        self.logger.info(message)  # Log the message with the standard logger
        formatted_message = self.string_handler.get_log()  # Capture the formatted log as text
        self.set_description_str(formatted_message)  # Set as tqdm description
        self.refresh()  # Immediately refresh to reflect changes

class LogStream(object):
    def __init__(self):
        self.logs = ''

    def write(self, str):
        self.logs += str

    def flush(self):
        pass

    def reset(self):
        self.logs = ''

    def __str__(self):
        return self.logs
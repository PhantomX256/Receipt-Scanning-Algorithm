import datetime
import os
import sys

DEBUG_LEVEL = "DEBUG"
INFO_LEVEL = "INFO"

class Logger:
    """
    A Lightweight stdout logger class
    Usage:
        logger = Logger()
        logger.info("Message")
        logger.debug("Message")
    """
    __slots__ = ("source_filename",)


    # Constructor that gets the filename
    def __init__(self):

        # Get current frame
        frame = sys._getframe(1)

        # Get the function name and set it
        self.source_filename = os.path.basename(frame.f_code.co_filename) if frame else "UNKNOWN"


    # Private method that logs the message along with level
    def __log(self, message: str, level: str) -> None:

        # Get the current frame
        frame = sys._getframe(2)

        # Get the function name
        function_name = frame.f_code.co_name if frame else "UNKNOWN"

        # Generate current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Log the message
        print(f"[{timestamp}]\t\t[{level}]\t\t{function_name}()@{self.source_filename}\t\t{message}")


    # DEBUG level log
    def debug(self, message: str) -> None:
        self.__log(message, DEBUG_LEVEL)


    # INFO level log
    def info(self, message: str) -> None:
        self.__log(message, INFO_LEVEL)

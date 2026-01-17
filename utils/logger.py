import datetime
import os
import sys


class Logger:
    """
    A Lightweight stdout logger class
    Usage:
        logger = Logger()
        logger.log("Log message")
    """
    __slots__ = ("source_filename",)

    def __init__(self):
        frame = sys._getframe(1)
        self.source_filename = os.path.basename(frame.f_code.co_filename) if frame else "UNKNOWN"

    def __log(self, message: str, level: str) -> None:
        frame = sys._getframe(2)
        function_name = frame.f_code.co_name if frame else "UNKNOWN"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}]\t\t[{level}]\t\t{function_name}()@{self.source_filename}\t\t{message}")

    def debug(self, message: str) -> None:
        self.__log(message, "DEBUG")

    def info(self, message: str) -> None:
        self.__log(message, "INFO")

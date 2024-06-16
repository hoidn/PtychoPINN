import os

class Configuration:
    def __init__(self, debug: bool = False, log_file_prefix: str = "logs"):
        self.debug = debug
        self.log_file_prefix = log_file_prefix

    def getDebugFlag(self) -> bool:
        return self.debug

    def getLogFilePrefix(self) -> str:
        return self.log_file_prefix


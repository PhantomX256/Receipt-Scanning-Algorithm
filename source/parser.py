from utils.logger import Logger


class Parser:
    """
    LLM Based parser that sends the output of OCR to an LLM and
    returns a JSON Output
    """
    def __init__(self):
        self.logger = Logger()

    def parse_text(self, ocr_text):

import os

import numpy as np
import pytesseract
from dotenv import load_dotenv

from utils.logger import Logger
from utils.timer import timer

load_dotenv()


class OCREngine:
    """
    A class to perform Optical Character Recognition (OCR) on images.
    Usage:
        ocr = OCREngine(language='eng')
        text = ocr.run_ocr(image)
    """
    # Tesseract engine config (change psm version)
    TESSERACT_CONFIG = r"--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.$:/-%,*#"

    OUTPUT_PATH = "./outputs/ocr_text/"

    # Get CMD path from env
    TESSERACT_CMD = os.getenv("TESSERACT_CMD")



    def __init__(self, language='eng'):
        self.language = language
        self.logger = Logger()


    def __write_to_file(self, text: str, input_path: str) -> None:

        # Ensure the directory exists
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)

        # Get filename
        base_name = os.path.basename(input_path)
        file_name_without_ext = os.path.splitext(base_name)[0]

        # Output file name
        output_file = os.path.join(self.OUTPUT_PATH, f"{file_name_without_ext}.txt")

        # Write output
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)


    @timer
    def run_ocr(self, image: np.ndarray, input_image_path: str) -> str:

        # Set CMD Path
        pytesseract.pytesseract.tesseract_cmd = self.TESSERACT_CMD

        # Do OCR
        ocr_text =  pytesseract.image_to_string(image, lang=self.language, config=self.TESSERACT_CONFIG)
        self.logger.info("Performed OCR")

        # Write the output
        self.__write_to_file(ocr_text, input_image_path)
        self.logger.info("OCR saved to outputs")

        return ocr_text
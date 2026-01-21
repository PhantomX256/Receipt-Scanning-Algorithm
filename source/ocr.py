import os

import numpy as np
import pytesseract
from dotenv import load_dotenv

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

    # Get CMD path from env
    TESSERACT_CMD = os.getenv("TESSERACT_CMD")


    def __init__(self, language='eng'):
        self.language = language


    def run_ocr(self, image: np.ndarray) -> str:

        # Set CMD Path
        pytesseract.pytesseract.tesseract_cmd = self.TESSERACT_CMD

        # Do OCR
        return pytesseract.image_to_string(image, lang=self.language, config=self.TESSERACT_CONFIG)
import os

import cv2
import numpy as np

from utils.logger import Logger
from utils.timer import timer
from rembg import remove


class Preprocessor:
    """
    A Preprocessor class.
    Usage:
        preprocessor = Preprocessor()
        preprocessor.run_preprocess(image_path)
    """
    # Default kwargs
    DEFAULT_MAX_DIM = 1800
    DEFAULT_CLAHE = False
    DEFAULT_INVERT = False

    OUTPUT_PATH = "./outputs/preprocessed_images/"

    # Initialize Logger
    logger = Logger()


    def __init__(self, max_dim: int=DEFAULT_MAX_DIM, clahe: bool=DEFAULT_CLAHE, invert: bool=DEFAULT_INVERT):
        self.max_dim = max_dim
        self.clahe = clahe
        self.invert = invert


    # Private method that caps dimensions
    def __cap_image_dim(self, image: np.ndarray) -> np.ndarray:

        # Get Dimensions
        height, width = image.shape[:2]

        # Calculate maximum dimension
        max_side = max(height, width)

        # If it is greater than max_dim then rescale
        if max_side > self.max_dim:
            scale = self.max_dim / float(max_side)
            image = cv2.resize(image, (int(width*scale), int(height*scale)), interpolation=cv2.INTER_AREA)

        return image


    # Private method to convert to 1 channel grayscale
    def __convert_to_gray(self, image):
        b, g, r, a = cv2.split(image)

        alpha_factor = a.astype(float) / 255.0

        composite = (b.astype(float) * alpha_factor) + (255.0 * (1.0 - alpha_factor))

        return composite.astype(np.uint8)

    # Private method that loads the input image and caps it at max_dim
    def  __load_image(self, image_path: str) -> np.ndarray:

        # Load the input image in grayscale from path using OpenCV
        loaded_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return self.__cap_image_dim(loaded_image)


    # Private method that applies gaussian blur to the image
    def __apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:

        # Apply Blur
        blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
        return blurred_image


    # Private method that applies CLAHE to the image
    def __apply_clahe(self, image: np.ndarray) -> np.ndarray:

        # Create CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # Apply to image
        return clahe.apply(image)


    # Private method that applies adaptive thresholding
    def __apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:

        # Binarize using adaptive Gaussian thresholding for uneven lighting
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )


    # Private method that applies morphological open and close
    def __apply_morphology(self, image: np.ndarray) -> np.ndarray:

        # Initialize kernel
        kernel = np.ones((2, 2), np.uint8)

        # Morphological Open
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

        # Morphological Close
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        return closed


    # Main public function that runs the preprocess pipeline
    @timer
    def run_preprocess(self, image_path: str) -> np.ndarray:

        # Load the image in grayscale
        preprocessed_image = self.__load_image(image_path)
        self.logger.info("Loaded Input Image")

        # Remove Background from image
        preprocessed_image = remove(preprocessed_image)
        self.logger.info("Removed Background")

        if preprocessed_image.shape[2] == 4:
            preprocessed_image = self.__convert_to_gray(preprocessed_image)
            self.logger.info("Image converted to grayscale")

        cv2.imshow("Preprocessed Image", preprocessed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Apply Gaussian Blur
        preprocessed_image = self.__apply_gaussian_blur(preprocessed_image)
        self.logger.info("Gaussian Blur Applied")

        # Apply CLAHE (If chosen)
        if self.clahe:
            preprocessed_image = self.__apply_clahe(preprocessed_image)
            self.logger.info("CLAHE Applied")

        # Apply adaptive thresholding
        preprocessed_image = self.__apply_adaptive_threshold(preprocessed_image)
        self.logger.info("Adaptive Threshold Applied")

        # Invert if requested (text bright on dark)
        if self.invert:
            preprocessed_image = cv2.bitwise_not(preprocessed_image)
            self.logger.info("Invert Applied")

        # Apply Morphological Open and Close
        preprocessed_image = self.__apply_morphology(preprocessed_image)
        self.logger.info("Morphology Applied")

        # Save Preprocessed image to path
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)
        cv2.imwrite(self.OUTPUT_PATH + os.path.basename(image_path), preprocessed_image)
        self.logger.info("Saved Preprocessed Image")

        self.logger.info("Preprocessing Complete")
        return preprocessed_image
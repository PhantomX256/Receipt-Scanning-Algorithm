import os

import cv2
import numpy as np
from rembg import remove

from utils.logger import Logger
from utils.timer import timer


class CropPipeline:
    """
    A pipeline to detect a document (receipt) in an image using edge detection
    and contour finding, then applying a perspective transform to obtain
    a top-down 'scanned' view of the document.
    """
    TARGET_DIM = 1080
    OUTPUT_PATH = "./outputs/cropped_images/"


    def __init__(self):
        self.logger = Logger()


    def __load_image(self, image_path: str) -> np.ndarray:
        """
        Loads the image at the path
        """
        # Load the input image from path using OpenCV
        return cv2.imread(image_path)


    def __resize_image(self, image):
        """
        Resize image to workable size
        """
        # Calculate the greatest dimension
        max_dim = max(image.shape)

        # Check if it exceeds the target
        if max_dim > self.TARGET_DIM:

            # Find scale to resize
            scale = self.TARGET_DIM / max_dim
            image = cv2.resize(image, None, fx=scale, fy=scale)

        return image


    def __apply_morphology(self, image: np.ndarray) -> np.ndarray:
        """
        Applies strong morphological operations (Close -> Open) to 'erase' text
        and details, leaving a reliable solid block for background extraction.
        """
        # A rectangular kernel
        kernel = np.ones((5, 5), np.uint8)

        # Morphological CLOSE
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)

        return image

    def __remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Removes background from the image using GrabCut.
        This isolates the receipt document from the table surface by looking at color distributions.
        """
        # Initialize a mas with same dimensions of image
        mask = np.zeros(image.shape[:2], np.uint8)

        # Reserve memory for algorithm's models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Define the Rectangle Recommendation
        rect = (20, 20, image.shape[1] - 20, image.shape[0] - 20)

        # Run GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Filter the Result
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Apply the mask
        image = image * mask2[:, :, np.newaxis]

        return image


    def __detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detects the outer edges of the document using Canny detection.
        Refines the edges with dilation to ensure the contour is closed.
        """
        # Convert image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply gaussian blur
        blurred_image = cv2.GaussianBlur(grayscale_image, (11, 11), 0)

        # Perform Edge detection
        edged_image = cv2.Canny(blurred_image, 0, 200)

        # Dilate the edges
        edges = cv2.dilate(edged_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        cv2.imshow("Edged Image", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return edges


    def __detect_contours(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Detects contours and returns the top 5 contours assuming receipt
        is in one of these contours
        """
        # Detect Contours
        contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Sort and return the top 5
        return sorted(contours, key=cv2.contourArea, reverse=True)[:5]


    def __detect_corners(self, contours: list[np.ndarray]) -> list:
        """
        Detects corners from the contours through approximation
        """
        # Iterate through each contour to find the corners
        for contour in contours:

            # Calculate perimeter for approximation
            epsilon = 0.02 * cv2.arcLength(contour, True)

            # Approximate the shape
            corners = cv2.approxPolyDP(contour, epsilon, True)

            # if the shape has 4 points then the box is found
            if len(corners) == 4:
                return sorted(np.concatenate(corners).tolist())

        # Flatten and sort
        return []


    def __order_points(self, corners: list):
        """
        Rearrange coordinates to order:
        top-left, top-right, bottom-right, bottom-left
        """
        # Create a container for 4 points
        rect = np.zeros((4, 2), dtype='float32')
        corners = np.array(corners)

        # Find Top-left and Bottom-Right
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]

        # Find Top-Right and Bottom-Left
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]

        # return the ordered coordinates
        return rect.astype('int').tolist()


    def __find_dest(self, points: np.ndarray) -> np.ndarray:
        """
        Calculates the dimensions of the transformed image based on the
        distances between the corners, and generates the target coordinates
        for the Perspective Transform.
        """
        # Unpack the ordered source points
        (tl, tr, br, bl) = points

        # Determine the maximum width
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Determine the maximum height
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Construct the destination points
        destination_corners = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        return destination_corners


    def __apply_transform(self, original_image: np.ndarray, source_corners: list, destination_corners: np.ndarray) -> np.ndarray:
        """
        Warps the original image based on the perspective transform from
        source_corners (receipt in photo) to destination_corners (flat document).
        """
        # Start with the "scanned" dimensions calculated in __find_dest
        max_width = int(destination_corners[2][0]) + 1
        max_height = int(destination_corners[2][1]) + 1

        # Get the Transform Matrix (M)
        M = cv2.getPerspectiveTransform(np.float32(source_corners), np.float32(destination_corners))

        # Warp the image
        return cv2.warpPerspective(original_image, M, (max_width, max_height), flags=cv2.INTER_LINEAR)


    @timer
    def crop_image(self, image_path):
        """
        Crop Pipeline
        """
        # Load the image
        input_image = self.__load_image(image_path)

        # Resize the input image
        input_image = self.__resize_image(input_image)

        # Keep a copy of the original image
        original_image = input_image.copy()

        # Use ML Based BG Remover and remove BG
        input_image = remove(input_image)

        cv2.imshow("BG Cropped", input_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Apply morphology
        input_image = self.__apply_morphology(input_image)

        # Detect Edges
        input_image = self.__detect_edges(input_image)

        # Detect Contours
        contours = self.__detect_contours(input_image)

        # Find Corners
        corners = self.__detect_corners(contours)

        if not corners:
            self.logger.info("No valid 4-corner receipt found")
            return original_image

        # Order the corners
        corners = self.__order_points(corners)

        # Final corners
        destination_corners = self.__find_dest(corners)

        input_image = self.__apply_transform(original_image, corners, destination_corners)

        os.makedirs(self.OUTPUT_PATH, exist_ok=True)
        cv2.imwrite(self.OUTPUT_PATH + os.path.basename(image_path), input_image)
        self.logger.info("Saved Preprocessed Image")

        return input_image

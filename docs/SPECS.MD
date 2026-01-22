# Receipt Scanning Pipeline - Specifications

## 1. Project Overview
This project is a Python-based receipt scanning pipeline. It takes an input image of a receipt, preprocesses it (optionally cropping or removing background), extracts text using Tesseract OCR, and parses the text into structured JSON using the Google Gemini API (LLM).

**Goal**: Convert receipt images into structured data (Merchant, Date, Items, Tax, Total) for free, using local OCR and the Gemini Flash API.

## 2. Technical Stack
- **Language**: Python 3.x
- **Computer Vision**: OpenCV (`cv2`), `rembg` (Background removal)
- **OCR Engine**: Tesseract (`pytesseract`)
  - *Note*: Tesseract binary must be installed on the system.
- **LLM**: Google Gemini (`google-genai`)
  - Model: `gemini-3-flash` (fallback: `gemini-1.5-flash` if 3.x not available).
- **Environment**: VS Code / Jupyter Notebook
- **OS**: Windows

## 3. Architecture & File Structure

The project is modularized into specific functional files:

### `source/crop.py`
- **Purpose**: Detect and crop the receipt document from a background.
- **Class**: `CropPipeline`
- **Method**: `crop_image(image_path)`
- **Logic**:
  - Load and resize image (target dim: 1080).
  - Remove background using `rembg`.
  - Detect edges (Canny), contours, and approximate corners.
  - Apply Perspective Transform to flatten the document.
  - Save cropped image to `outputs/cropped_images/`.

### `source/preprocess.py`
- **Purpose**: Prepare the image for OCR to maximize accuracy.
- **Class**: `Preprocessor`
- **Method**: `run_preprocess(image_path)`
- **Logic**:
  - Load image using `cv2` (handles alpha channel if present).
  - Remove background using `rembg`.
  - Convert to Grayscale.
  - Apply Gaussian Blur (noise reduction).
  - Optionally apply CLAHE (local contrast) when lighting is uneven.
  - Apply Adaptive Thresholding (binarization), invert if needed.
  - Optionally apply morphological clean-up (open/close).
  - Save processed image to `outputs/preprocessed_images/`.

### `source/ocr.py`
- **Purpose**: Extract raw text from the image.
- **Class**: `OCREngine`
- **Method**: `run_ocr(image, input_image_path)`
- **Logic**:
  - Check for `TESSERACT_CMD` in environment variables.
  - Configure Tesseract (PSM 6 - Single uniform block of text).
  - Run `image_to_string`.
  - Save raw text to `outputs/ocr_text/`.
  - Return: Raw extracted string.

### `source/parser.py`
- **Purpose**: Structure the raw text into JSON using LLM.
- **Class**: `Parser`
- **Status**: **Skeleton / Pending Implementation**.
- **Method**: `parse_text(ocr_text)`
- **Logic**:
  - Initialize Gemini API with `GEMINI_API_KEY`.
  - Construct a prompt including the OCR text and schema requirements.
  - Request JSON response.
  - Clean and parse JSON.
  - Return: Dictionary object.

### `main.ipynb` (Orchestrator)
- **Purpose**: User interface to run the pipeline.
- **Steps**:
  1. Import `Preprocessor` and `OCREngine`.
  2. Initialize instances.
  3. Define input image path.
  4. Run `preprocessor.run_preprocess()`.
  5. Run `ocr_engine.run_ocr()`.

## 4. Common Utilities

### `utils/logger.py`
- **Purpose**: Lightweight stdout logger.
- **Class**: `Logger`
- **Usage**:
  - Instantiate `logger = Logger()`.
  - Use `logger.debug(msg)` or `logger.info(msg)`.
  - Logger auto-captures caller function and source filename.

### `utils/timer.py`
- **Purpose**: Performance monitoring decorator.
- **Usage**: Decorate functions with `@timer` to log execution time.

## 5. Output Data Model
The output must be a JSON object with this exact structure:

```json
{
  "merchant_name": "String (e.g., Walmart)",
  "date": "String (YYYY-MM-DD)",
  "items": [
    {
      "item_name": "String",
      "quantity": "Number (default 1)",
      "price": "Number"
    }
  ],
  "tax_percentage": "Number (null if not found)",
  "final_total": "Number (inclusive of tax)"
}
```

## 6. Configuration
- **Environment Variables** (`.env`):
  - `GEMINI_API_KEY`: API key for Google Gemini.
  - `TESSERACT_CMD`: Full path to `tesseract.exe`.
- **Dependencies**: Listed in `requirements.txt`.

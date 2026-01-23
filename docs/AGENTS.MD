# Receipt Scanning Pipeline - Agent Context & Specifications

## 1. Project Overview
This project is a Python-based receipt scanning pipeline. It takes an input image of a receipt, preprocesses it (optionally cropping or removing background), extracts text using Tesseract OCR, and parses the text into structured JSON using the Google Gemini API (LLM).

**Goal**: Convert receipt images into structured data (Merchant, Date, Items, Tax, Total) for free, using local OCR and the Gemini Flash API.

**Status**: ✅ Fully Implemented (January 2026)

## 2. Technical Stack
- **Language**: Python 3.x
- **Computer Vision**: OpenCV (`cv2`), `rembg` (ML-based background removal)
- **OCR Engine**: Tesseract (`pytesseract`)
  - *Note*: Tesseract binary must be installed on the system and path set via `TESSERACT_CMD` env variable.
- **LLM**: Google Gemini (`google-genai` SDK)
  - Model: `gemini-2.5-flash` (supports thinking/reasoning mode)
  - Features: JSON mode, safety filters disabled for OCR noise, high token limit (8192) for reasoning models
- **Environment**: VS Code / Jupyter Notebook
- **OS**: Windows

## 3. Architecture & File Structure

The project is modularized into specific functional files:

### `source/crop.py`
- **Purpose**: Detect and crop the receipt document from a background using computer vision.
- **Class**: `CropPipeline`
- **Configuration**:
  - `TARGET_DIM = 1080` (resize images to manageable dimension)
  - `OUTPUT_PATH = "./outputs/cropped_images/"`
- **Method**: `crop_image(image_path)` (decorated with `@timer`)
- **Pipeline Steps**:
  1. `__load_image()` - Load image with OpenCV
  2. `__resize_image()` - Cap max dimension to 1080px for performance
  3. `remove()` - ML-based background removal using `rembg`
  4. `__apply_morphology()` - Apply morphological close operation (kernel 5x5, 3 iterations) to erase text details
  5. `__detect_edges()` - Grayscale → Gaussian Blur (11x11) → Canny edge detection (0, 200) → Dilation (ellipse 5x5)
  6. `__detect_contours()` - Find top 5 contours by area using `cv2.findContours(RETR_LIST, CHAIN_APPROX_NONE)`
  7. `__detect_corners()` - Approximate polygons with epsilon=0.02 to find 4-corner quadrilaterals
  8. `__order_points()` - Arrange corners as [top-left, top-right, bottom-right, bottom-left]
  9. `__find_dest()` - Calculate output dimensions based on corner distances (width and height)
  10. `__apply_transform()` - Apply `cv2.getPerspectiveTransform()` and warp to flat document
  11. Save to `outputs/cropped_images/`
- **Alternative Method**: `__remove_background()` using GrabCut algorithm (available but not in active pipeline)
- **Debug Features**: Shows intermediate images with `cv2.imshow()` for edge detection and background removal validation

### `source/preprocess.py`
- **Purpose**: Prepare the image for OCR to maximize text extraction accuracy.
- **Class**: `Preprocessor`
- **Configuration**:
  - `DEFAULT_MAX_DIM = 1800` (caps image size)
  - `DEFAULT_CLAHE = False` (contrast enhancement toggle)
  - `DEFAULT_INVERT = False` (invert black/white toggle)
  - `DEFAULT_MORPH = False` (morphological operations toggle)
  - `OUTPUT_PATH = "./outputs/preprocessed_images/"`
- **Constructor**: `__init__(max_dim, clahe, invert, morph)` - All parameters optional with defaults
- **Method**: `run_preprocess(image_path)` (decorated with `@timer`)
- **Pipeline Steps**:
  1. `__load_image()` - Load in grayscale with `cv2.IMREAD_GRAYSCALE`
  2. `__cap_image_dim()` - Resize if max side > max_dim using `cv2.INTER_AREA`
  3. `remove()` - Background removal using `rembg` library
  4. `__convert_to_gray()` - Handle RGBA (4-channel) images by compositing alpha channel: `composite = (b * alpha_factor) + (255 * (1 - alpha_factor))`
  5. `__apply_gaussian_blur()` - (3x3) kernel for noise reduction
  6. `__apply_clahe()` (optional) - Contrast Limited Adaptive Histogram Equalization (clipLimit=2.0, tileGridSize=8x8)
  7. `__apply_adaptive_threshold()` - Gaussian adaptive method (block size=31, C=10, THRESH_BINARY)
  8. Invert (optional) - `cv2.bitwise_not()` for bright text on dark background
  9. `__apply_morphology()` (optional) - Open (remove noise) then Close (fill gaps) operations with 2x2 kernel
  10. Save to `outputs/preprocessed_images/`

### `source/ocr.py`
- **Purpose**: Extract raw text from the preprocessed image using Tesseract OCR.
- **Class**: `OCREngine`
- **Configuration**:
  - `TESSERACT_CONFIG = "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.$:/-%,*#"`
    - OEM 1: LSTM neural net mode
    - PSM 6: Assume uniform block of text
    - Whitelist: Receipt-specific characters only (letters, numbers, common symbols)
  - `OUTPUT_PATH = "./outputs/ocr_text/"`
  - `TESSERACT_CMD`: Loaded from `.env` file
- **Constructor**: `__init__(language='eng')` - Sets OCR language
- **Method**: `run_ocr(image: np.ndarray, input_image_path: str) -> str` (decorated with `@timer`)
- **Logic**:
  1. Set Tesseract command path from environment variable
  2. Execute `pytesseract.image_to_string()` with configured parameters
  3. `__write_to_file()` - Write raw OCR text to `outputs/ocr_text/<filename>.txt`
  4. Return raw text string

### `source/parser.py`
- **Purpose**: Parse raw OCR text into structured JSON using Google Gemini LLM.
- **Class**: `Parser`
- **Status**: ✅ **Fully Implemented**
- **Configuration**:
  - `GEMINI_API_KEY`: Loaded from `.env` via `os.getenv()`
  - `MODEL_ID = "gemini-2.5-flash"` (supports thinking/reasoning mode)
  - `STRUCTURE_FILE_PATH = "output_structure.txt"` (JSON schema definition)
  - `OUTPUT_PATH = "./outputs/json/"`
  - `PROMPT_TEMPLATE`: Multi-line template with extraction instructions
  - `GENERATE_CONTENT_CONFIG`:
    - `response_mime_type="application/json"` - Forces JSON output format
    - `max_output_tokens=8192` - High limit to accommodate model's internal reasoning tokens
    - `temperature=0.1` - Low temperature for deterministic output
    - `safety_settings`: Embedded in config
  - `SAFETY_SETTINGS`: All harm categories set to `BLOCK_NONE` to prevent OCR noise from triggering safety filters:
    - `HARM_CATEGORY_HATE_SPEECH → BLOCK_NONE`
    - `HARM_CATEGORY_SEXUALLY_EXPLICIT → BLOCK_NONE`
    - `HARM_CATEGORY_DANGEROUS_CONTENT → BLOCK_NONE`
    - `HARM_CATEGORY_HARASSMENT → BLOCK_NONE`
- **Client Initialization**: `genai.Client(api_key=GEMINI_API_KEY)`
- **Method**: `parse_text(ocr_text: str, input_image_path: str) -> dict`
- **Private Methods**:
  - `__get_structure() -> str`: Reads `output_structure.txt` and returns schema as string for prompt injection
  - `__save_json(json_output: dict, input_image_path: str) -> None`: Saves parsed JSON to `outputs/json/<filename>.json` using original filename
- **Processing Logic**:
  1. **Prompt Construction**: Inject OCR text and structure schema into `PROMPT_TEMPLATE`
  2. **API Call**: `client.models.generate_content()` with model ID, prompt, and config
  3. **Response Validation**: Check if response text exists, raise error if empty
  4. **JSON Parsing**: `json.loads(response.text)` to convert string to dict
  5. **Save**: Call `__save_json()` to persist result to disk
  6. **Error Handling**: Catch all exceptions and return error dict with snippet of OCR text
- **Prompt Instructions to LLM**:
  - Analyze OCR text and correct common mistakes (e.g., 'S' → '5', 'O' → '0')
  - Return `null` for fields that cannot be inferred with confidence
  - Do not make up data
  - Return only JSON (no markdown formatting)
- **Error Response Format**: `{"error": "Parsing failed", "details": str(e), "raw_ocr_text_snippet": ocr_text[:100]}`

### `main.ipynb` (Orchestrator)
- **Purpose**: User interface to run the full pipeline.
- **Cells**:
  1. **Imports & Initialization**: Import all classes (`Preprocessor`, `OCREngine`, `Parser`) and instantiate
  2. **Input Configuration**: Set `input_image_path` variable (e.g., `"dataset/walmart_receipts/11.jpg"`)
  3. **Preprocessing**: `preprocessed_image = preprocessor.run_preprocess(input_image_path)`
  4. **OCR**: `ocr_text = ocr_engine.run_ocr(preprocessed_image, input_image_path)`
  5. **Parsing**: `json = parser.parse_text(ocr_text, input_image_path)`
- **Execution**: Run cells sequentially in Jupyter notebook

## 4. Common Utilities

### `utils/logger.py`
- **Purpose**: Lightweight stdout logger with semantic information.
- **Class**: `Logger`
- **Implementation Details**:
  - Uses `__slots__ = ("source_filename",)` for memory efficiency
  - Auto-captures calling function name via `sys._getframe(2).f_code.co_name`
  - Auto-captures source filename via `sys._getframe(1).f_code.co_filename` during `__init__()`
  - Timestamp format: `%Y-%m-%d %H:%M:%S`
  - Output format: `[timestamp]\t\t[LEVEL]\t\t{function_name}()@{filename}\t\t{message}`
- **Methods**:
  - `__init__()`: Captures source filename from caller's frame
  - `info(message: str)`: Logs at INFO level
  - `debug(message: str)`: Logs at DEBUG level
  - `__log(message: str, level: str)`: Private method that handles formatting and printing
- **Usage**:
```python
logger = Logger()
logger.info("Processing started")
logger.debug("Debug information")
```

### `utils/timer.py`
- **Purpose**: Performance monitoring decorator.
- **Decorator**: `@timer`
- **Implementation**:
  - Uses `functools.wraps(function)` to preserve function metadata
  - Measures execution time with `time.perf_counter()` for high-resolution timing
  - Creates `Logger` instance inside wrapper
  - Logs via `Logger.info()` with format: `"Finished {function_name}() in {run_time:.4f} secs"`
- **Usage**:
```python
@timer
def my_function():
    # function code will be timed automatically
```

## 5. Output Data Model
Schema defined in `output_structure.txt` and enforced by LLM:
```json
{
  "merchant_name": "String (name of merchant/store, or null if not found)",
  "date": "String (YYYY-MM-DD format, or null if not found)",
  "items": [
    {
      "name": "String (item name)",
      "quantity": "Integer (quantity of item)",
      "price": "Float (price per unit)"
    }
  ],
  "total_price": "Float (total including tax, no currency symbol)",
  "tax_percentage": "Float (tax rate as percentage, or null if not found)"
}
```
**Graceful Degradation**: All fields can be `null` if the LLM cannot extract or infer the information with confidence. This prevents hallucinated data.

## 6. Configuration
- **Environment Variables** (`.env` file in project root):
  - `GEMINI_API_KEY`: Google Gemini API key (required for `parser.py`)
  - `TESSERACT_CMD`: Full Windows path to `tesseract.exe` (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`)
- **Dependencies**: Listed in `requirements.txt`
  - Key packages: `opencv-python`, `pytesseract`, `google-genai`, `rembg`, `python-dotenv`, `numpy`

## 7. Development Instructions
- Use the `Logger` for all console output instead of `print()`.
- Decorate long-running functions with `@timer` for performance monitoring.
- All modules are fully implemented and functional.
- Check `outputs/` directories for intermediate results during debugging.
- Parser returns error dict on failure with `error`, `details`, and `raw_ocr_text_snippet` keys.
- Review logs with timestamps and function names for troubleshooting.

## 8. Known Issues & Solutions

### Issue: Incomplete JSON Response from Gemini (MAX_TOKENS)
- **Symptom**: JSON cuts off mid-generation (e.g., `"name":` without value)
- **Cause**: `gemini-2.5-flash` uses internal reasoning tokens (`thoughts_token_count`) that count against the output token limit. For example, it may use 1917 tokens for reasoning, leaving only ~80 tokens for the actual JSON output.
- **Solution**: Set `max_output_tokens=8192` in `GenerateContentConfig` to accommodate both reasoning and output
- **Debug**: Check `response.candidates[0].finish_reason` - if `MAX_TOKENS`, increase the limit

### Issue: Safety Filter False Positives
- **Symptom**: Empty or truncated response from Gemini
- **Cause**: Garbled OCR text (e.g., "FrozenHensoes") can trigger safety filters for hate speech, harassment, or sexually explicit content
- **Solution**: All safety settings set to `BLOCK_NONE` in config (already implemented)
- **Debug**: Check `response.candidates[0].finish_reason` - if `SAFETY`, review `safety_ratings` for triggered categories

### Issue: RGBA Image Handling After Background Removal
- **Symptom**: 4-channel image after `rembg.remove()` causes grayscale conversion issues
- **Cause**: Background removal adds an alpha channel, creating RGBA format
- **Solution**: `__convert_to_gray()` in `preprocess.py` manually composites alpha channel with formula: `composite = (channel * alpha_factor) + (255 * (1 - alpha_factor))`
- **Already Implemented**: Code checks `if preprocessed_image.shape[2] == 4` and converts automatically

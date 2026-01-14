# Receipt Scanning Pipeline - Agent Context & Specifications

## 1. Project Overview
This project is a Python-based receipt scanning pipeline. It takes an input image of a receipt, preprocesses it, extracts text using Tesseract OCR, and parses the text into structured JSON using the Google Gemini API (LLM).

**Goal**: Convert receipt images into structured data (Merchant, Date, Items, Tax, Total) for free, using local OCR and the Gemini Flash API.

## 2. Technical Stack
- **Language**: Python 3.x
- **Computer Vision**: OpenCV (`cv2`)
- **OCR Engine**: Tesseract (`pytesseract`)
  - *Note*: Tesseract binary must be installed on the system.
- **LLM**: Google Gemini (`google-generativeai`)
  - Model: `gemini-1.5-flash` (or `gemini-3-flash` if available/configured).
- **Environment**: VS Code / Jupyter Notebook
- **OS**: Windows

## 3. Architecture & File Structure

The project is modularized into specific functional files:

### `preprocess.py`
- **Purpose**: Prepare the image for OCR.
- **Function**: `run_preprocess(input_image)`
- **Logic**:
  - Load image (path or array).
  - Convert to Grayscale.
  - Apply Gaussian Blur (noise reduction).
  - Apply Adaptive Thresholding (binarization).
  - Return: The processed image (numpy array).

### `ocr.py`
- **Purpose**: Extract raw text from the image.
- **Function**: `run_ocr(processed_image)`
- **Logic**:
  - Check for `TESSERACT_CMD` in environment variables.
  - Configure Tesseract (PSM 6 - Single uniform block of text).
  - Run `image_to_string`.
  - Return: Raw extracted string.

### `parser.py`
- **Purpose**: Structure the raw text into JSON.
- **Function**: `extract_receipt_info(ocr_text, model_name)`
- **Logic**:
  - Initialize Gemini API with `GEMINI_API_KEY`.
  - Construct a prompt including the OCR text and schema requirements.
  - Request JSON response.
  - Clean and parse JSON.
  - Return: Dictionary object.

### `main.ipynb` (Orchestrator)
- **Purpose**: User interface to run the pipeline.
- **Steps**:
  1. Define Image Path.
  2. Call `run_preprocess`.
  3. Call `run_ocr`.
  4. Call `extract_receipt_info`.
  5. Display/Save results.

## 4. Output Data Model
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

## 5. Configuration
- **Environment Variables** (`.env`):
  - `GEMINI_API_KEY`: API key for Google Gemini.
  - `TESSERACT_CMD`: (Optional) Full path to `tesseract.exe` if not in PATH.
- **Dependencies**: Listed in `requirements.txt`.

## 6. Development Instructions for Agents
- Keep code simple and modular.
- Use `try-except` blocks for robustness.
- Do not overcomplicate the image processing (start with standard adaptive thresholding).
- Ensure the LLM prompt enforces strictly valid JSON.


# Receipt Scanning Pipeline

This project converts receipt images into structured JSON using a local OCR-plus-LLM workflow.

The current implementation uses:

1. Document detection and perspective correction with `processor.scan.DocScanner`
2. OCR extraction with Surya
3. Receipt text reconstruction from OCR bounding boxes
4. Structured parsing with a local Ollama-hosted Gemma model

The result is a JSON file containing receipt fields such as store name, date, line items, subtotal, tax, and total.

## Overview

The notebook entry point is `main.ipynb`. A typical run performs the following steps:

1. Load a receipt image from `dataset/images/`
2. Crop and deskew the receipt using `DocScanner`
3. Run Surya OCR and save the raw OCR output to `output/ocr/`
4. Reconstruct receipt text using OCR line bounding boxes
5. Send the reconstructed text to Ollama with `gemma4:e4b`
6. Validate and save the final structured result to `output/json/`

## Features

- Receipt detection and perspective correction
- OCR output preservation for debugging and traceability
- Layout-aware reconstruction of OCR text before LLM parsing
- Local LLM parsing through Ollama
- JSON validation using a typed schema
- Per-receipt outputs stored in predictable folders

## Requirements

- Python 3.10 or later is recommended
- macOS users may need Qt support for the interactive cropping window
- Ollama installed locally
- The `gemma4:e4b` model pulled into Ollama

If you are using the interactive crop window in Jupyter, make sure the Qt backend is available in your environment.

## Installation

1. Clone the repository

```bash
git clone <repository-url>
cd Receipt-Scanning
```

2. Install Python dependencies

```bash
pip install -r requirements.txt
```

3. Install or start Ollama

```bash
ollama pull gemma4:e4b
```

If the model is already installed, you can verify it with:

```bash
ollama list
```

## Usage

### Notebook workflow

Open `main.ipynb` and run the cells in order.

The current pipeline is:

```python
processed_receipt = scanner.scan(INPUT_IMAGE_PATH)
formatted_text = run_surya_ocr(processed_receipt, filename)
final_json = run_receipt_llm_parsing(formatted_text, filename)
```

### Inputs

Place receipt images in `dataset/images/`.

### Outputs

Each receipt produces:

- `output/processed/<filename>.jpg` - cropped and thresholded receipt image
- `output/ocr/<filename>.json` - Surya OCR output
- `output/ocr/<filename>.txt` - reconstructed OCR text
- `output/json/<filename>.json` - final structured receipt JSON

## Project Structure

```text
Receipt-Scanning/
├── main.ipynb
├── requirements.txt
├── README.MD
├── dataset/
│   └── images/
├── output/
│   ├── processed/
│   ├── ocr/
│   └── json/
├── processor/
│   ├── scan.py
│   └── polygon_interacter.py
├── ocr_parser/
│   ├── parser.py
│   └── __init__.py
└── llm_parser/
    ├── parser.py
    ├── json_format.py
    └── __init__.py
```

## How It Works

### 1. Document scanning and preprocessing

`processor/scan.py` contains `DocScanner`, which detects the receipt boundary, corrects perspective, and produces a cleaned, thresholded image.

### 2. OCR extraction

`ocr_parser/parser.py` uses Surya OCR to extract text lines and bounding boxes from the processed receipt image. The raw OCR result is saved as JSON for inspection.

### 3. Text reconstruction

The OCR JSON is converted into a line-oriented text representation using bounding box coordinates. This preserves layout information that is important for receipts.

### 4. LLM parsing

`llm_parser/parser.py` sends the reconstructed OCR text to Ollama using `gemma4:e4b`. The response is validated against the receipt schema defined in `llm_parser/json_format.py` and written to disk as JSON.

## JSON Schema

The structured receipt output is defined in `llm_parser/json_format.py`. The current schema is intended to represent the following receipt fields:

- store name
- date
- list of items
- subtotal
- tax
- total

The LLM output is validated before it is saved.

## Troubleshooting

### Ollama model does not load

- Confirm Ollama is running locally
- Run `ollama list` to verify `gemma4:e4b` is installed
- If needed, pull the model again with `ollama pull gemma4:e4b`

### OCR output looks incorrect

- Try a clearer input image
- Use the interactive crop window to adjust the receipt boundary
- Check the contents of `output/ocr/<filename>.json` and `output/ocr/<filename>.txt`

### LLM output is empty or incomplete

- Verify that the reconstructed OCR text is not empty
- Review the prompt used in `llm_parser/parser.py`
- Confirm the local model is responding correctly through Ollama

## Notes

- This project is designed to run locally.
- The raw OCR output is kept intentionally so the parsing pipeline is easy to debug.
- The output format may evolve as the schema in `llm_parser/json_format.py` changes.

## License

This project is provided for educational and personal use.

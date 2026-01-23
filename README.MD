# 🧾 Receipt Scanning Algorithm

A Python-based receipt scanning pipeline that converts receipt images into structured JSON data using OCR and LLM technology.

## 📋 Overview

This project processes receipt images through a multi-stage pipeline:
1. **Preprocessing** - Cleans and enhances images for optimal OCR performance
2. **OCR Extraction** - Extracts raw text using Tesseract OCR
3. **LLM Parsing** - Structures the text into JSON using Google Gemini AI

**Result**: Clean, structured data containing merchant name, date, items, prices, tax, and totals.

## ✨ Features

- 🖼️ **Image Preprocessing** - Background removal, contrast enhancement, noise reduction, and binarization
- 📐 **Document Cropping** - Automatic receipt detection and perspective correction (optional)
- 🔍 **OCR Extraction** - Tesseract-based text extraction with custom configurations
- 🤖 **AI Parsing** - Google Gemini LLM converts messy OCR text into clean JSON
- 🛡️ **Graceful Degradation** - Returns `null` for fields that cannot be extracted with confidence
- 📊 **Structured Output** - Consistent JSON format for easy integration

## 🚀 Quick Start

### Prerequisites

1. **Python 3.x** installed
2. **Tesseract OCR** installed on your system
   - Download: https://github.com/tesseract-ocr/tesseract
   - Windows: Install and note the installation path (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`)
3. **Google Gemini API Key**
   - Get one at: https://ai.google.dev/

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Receipt Scanning Algorithm"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

### Usage

1. **Open the Jupyter notebook**
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Set your input image path**
   ```python
   input_image_path = "dataset/walmart_receipts/11.jpg"
   ```

3. **Run all cells sequentially**
   - The notebook will preprocess, OCR, and parse your receipt
   - Check the `outputs/` folder for results

## 📁 Project Structure

```
Receipt Scanning Algorithm/
├── main.ipynb                 # Main pipeline orchestrator
├── output_structure.txt       # JSON schema definition
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
│
├── source/                    # Core modules
│   ├── preprocess.py          # Image preprocessing
│   ├── ocr.py                 # OCR text extraction
│   ├── parser.py              # LLM-based JSON parsing
│   └── crop.py                # Document detection & cropping (optional)
│
├── utils/                     # Utility modules
│   ├── logger.py              # Logging utility
│   └── timer.py               # Performance timing decorator
│
├── dataset/                   # Input receipt images
│   ├── walmart_receipts/
│   ├── restaurant_receipts/
│   └── other_receipts/
│
├── outputs/                   # Generated outputs
│   ├── preprocessed_images/   # Cleaned images ready for OCR
│   ├── ocr_text/              # Raw text files from Tesseract
│   └── json/                  # Final structured JSON outputs
│
└── docs/                      # Documentation
    ├── AGENTS.MD              # Technical specifications
    └── SPECS.MD               # Project specifications
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key for LLM parsing | `AIzaSy...` |
| `TESSERACT_CMD` | Full path to Tesseract executable | `C:\Program Files\Tesseract-OCR\tesseract.exe` |

### Preprocessing Options

You can customize the `Preprocessor` class with these parameters:

```python
preprocessor = Preprocessor(
    max_dim=1800,      # Maximum image dimension
    clahe=False,       # Contrast Limited Adaptive Histogram Equalization
    invert=False,      # Invert black/white (for dark backgrounds)
    morph=False        # Apply morphological operations
)
```

### OCR Configuration

The OCR engine uses these Tesseract settings:
- **OEM 1**: LSTM neural net mode
- **PSM 6**: Assume uniform block of text
- **Character whitelist**: Letters, numbers, and common receipt symbols

## 📤 Output Format

The pipeline generates JSON files with this structure:

```json
{
  "merchant_name": "Walmart",
  "date": "2026-01-15",
  "items": [
    {
      "name": "PLTORTILLAS",
      "quantity": 1,
      "price": 6.99
    },
    {
      "name": "CAGEFREEALLWHIT",
      "quantity": 1,
      "price": 3.69
    }
  ],
  "total_price": 12.45,
  "tax_percentage": 8.5
}
```

**Note**: Fields will be `null` if the information cannot be extracted with confidence.

## 📂 Output Locations

All outputs are saved in the `outputs/` directory with the same base filename as the input image:

- **Preprocessed Images**: `outputs/preprocessed_images/11.jpg`
- **OCR Text**: `outputs/ocr_text/11.txt`
- **JSON Data**: `outputs/json/11.json`

## 🔍 How It Works

### 1. Image Preprocessing (`source/preprocess.py`)
- Removes background using ML-based `rembg` library
- Converts to grayscale and handles alpha channels
- Applies Gaussian blur for noise reduction
- Optional CLAHE for contrast enhancement
- Adaptive thresholding for binarization
- Optional morphological operations

### 2. OCR Extraction (`source/ocr.py`)
- Uses Tesseract OCR with custom configuration
- Character whitelist for receipt-specific symbols
- Saves raw text output for debugging

### 3. LLM Parsing (`source/parser.py`)
- Sends OCR text to Google Gemini (`gemini-2.5-flash`)
- Uses structured prompt with schema definition
- JSON mode enabled for reliable output
- Safety filters disabled to handle OCR noise
- High token limit (8192) for reasoning models
- Graceful error handling with partial results

## 🛠️ Troubleshooting

### Common Issues

**Issue**: "Tesseract not found"
- **Solution**: Make sure Tesseract is installed and `TESSERACT_CMD` in `.env` points to the correct path

**Issue**: Empty JSON response from Gemini
- **Solution**: Check your `GEMINI_API_KEY` in `.env` is valid and has quota remaining

**Issue**: Incomplete JSON (cuts off mid-generation)
- **Solution**: Already handled! The code uses `max_output_tokens=8192` to accommodate reasoning tokens

**Issue**: Poor OCR results
- **Solution**: 
  - Try enabling CLAHE: `Preprocessor(clahe=True)`
  - For dark backgrounds: `Preprocessor(invert=True)`
  - Ensure input image is clear and well-lit

## 📊 Performance

Typical processing times (may vary based on hardware):
- **Preprocessing**: 2-5 seconds
- **OCR Extraction**: 1-3 seconds
- **LLM Parsing**: 5-15 seconds (depends on API latency)

**Total**: ~10-25 seconds per receipt

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

## 📄 License

This project is provided as-is for educational and personal use.

## 🙏 Acknowledgments

- **Tesseract OCR** - Open-source OCR engine
- **Google Gemini** - AI-powered text parsing
- **OpenCV** - Image processing
- **rembg** - ML-based background removal

---

**Last Updated**: January 23, 2026  
**Status**: ✅ Fully Functional

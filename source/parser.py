import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from utils.logger import Logger

load_dotenv()


class Parser:
    """
    LLM Based parser that sends the output of OCR to an LLM and
    returns a JSON Output
    """
    # Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_ID = "gemini-2.5-flash"
    STRUCTURE_FILE_PATH = "output_structure.txt"
    OUTPUT_PATH = "./outputs/json/"
    PROMPT_TEMPLATE = """
    You are an information extraction system. Your task is to extract structured receipt data from noisy OCR text.

    Return ONLY valid JSON that conforms exactly to the provided schema. Do not include markdown, explanations, or additional keys.
    
    IMPORTANT RULES:
    - Be conservative: do not invent data. Use null for any field you cannot infer with high confidence.
    - Correct common OCR mistakes when obvious (e.g., O→0, S→5, B→8, missing decimal points).
    - All currency values must be numbers only (no $ signs). Use decimals when applicable.
    - If quantity is not explicitly shown for an item, set quantity to 1.
    - Items must represent PURCHASED products/menu items only.
    
    EXCLUDE these lines from items (do NOT include them as items):
    - tips / suggested tips / gratuity / service charge / delivery fee
    - discounts / coupons / promotions / savings (including negative-priced lines)
    - payment lines (cash, card, VISA, Mastercard, change due)
    - store headers/footers, addresses, phone numbers, transaction ids, cashier ids
    - tax lines (tax is captured separately)
    
    ITEM PRICING REQUIREMENTS:
    - Each item must include:
      - price_per_unit: the unit price when available
      - total_price: the line total for that item line when available
    - If the OCR shows patterns like "2 @ 1.50 3.00" or "2x 1.50 = 3.00":
      - quantity = 2
      - price_per_unit = 1.50
      - total_price = 3.00
    - If only a single price is present on an item line and quantity is 1, it is acceptable to set:
      - price_per_unit = that price
      - total_price = that same price
    - If you cannot determine one of price_per_unit or total_price, set that field to null (do not guess).
    
    TOTAL AND TAX REQUIREMENTS:
    - total_price at the root level must be the GRAND TOTAL / final amount due (including tax).
      Prefer labels like: GRAND TOTAL, TOTAL DUE, AMOUNT DUE, BALANCE DUE, TOTAL.
    - tax_percentage must be the effective combined tax rate (percentage).
      - If multiple taxes exist, sum the tax amounts and infer ONE effective percentage when possible.
      - If an explicit percentage is printed, you may use it.
      - If you cannot infer a reliable percentage, set tax_percentage to null.
    
    DATE REQUIREMENTS:
    - date must be formatted as YYYY-MM-DD.
    - If the date cannot be confidently determined, return null.
    
    SCHEMA (must match exactly):
    {STRUCTURE_SCHEMA}
    
    OCR TEXT:
    ```
    {OCR_TEXT}
    ```
    """
    SAFETY_SETTINGS = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        )
    ]
    GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
        response_mime_type="application/json",
        max_output_tokens=8192,
        temperature=0.1,
        safety_settings=SAFETY_SETTINGS
    )

    # Initialize Logger
    logger = Logger()

    # Initialize client
    client = genai.Client(api_key=GEMINI_API_KEY)

    def __get_structure(self) -> str:
        """
        Reads the output_structure.txt file to be used in the prompt.
        Handles path resolution assuming file is in project root.
        """
        with open(self.STRUCTURE_FILE_PATH, 'r', encoding='utf-8') as file:
            return file.read()

    def __save_json(self, json_output: dict, input_image_path: str) -> None:
        """
        Takes the JSON Ouput and input_image_path and saves the ouput as a json in outputs
        with the original filename
        """
        # Ensure directory exists or make it
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)

        # Get the file name from path and use it
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        save_path = os.path.join(self.OUTPUT_PATH, base_name + ".json")

        # Save the json
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4)

    def parse_text(self, ocr_text: str, input_image_path: str) -> dict:
        """
        Takes the OCR Text and passes it to an LLM to process
        into a JSON output
        """
        # Prepare the prompt
        self.logger.info("Preparing prompt for Gemini...")
        structure_content = self.__get_structure()
        prompt = self.PROMPT_TEMPLATE.format(
            structure_content=structure_content,
            ocr_text=ocr_text
        )

        self.logger.info("Sending OCR text to Gemini")
        # Send the prompt and text to gemini
        try:
            response = self.client.models.generate_content(
                model=self.MODEL_ID,
                contents=prompt,
                config=self.GENERATE_CONTENT_CONFIG,
            )

            # If no response is received
            if not response.text:
                raise ValueError("Received an empty response")

            self.logger.info("Received a response from Gemini. Parsing...")

            # print(response)

            # Parse the json from text
            parsed_json = json.loads(response.text)
            self.logger.info("Successfully parsed receipt data")

            # Save JSON
            self.__save_json(parsed_json, input_image_path)
            self.logger.info("Saved JSON to outputs")

            return parsed_json

        except Exception as e:
            self.logger.info("Error during LLM Parsing")
            return {
                "error": "Parsing failed",
                "details": str(e),
                "raw_ocr_text_snippet": ocr_text[:100]
            }

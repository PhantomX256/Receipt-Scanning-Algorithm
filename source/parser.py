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
    You are an expert receipt parsing agent. Your task is to extract structured data from the raw OCR text of a receipt.

    STRICTLY follow the structure defined below for your output:

    {structure_content}

    Instructions:
    1. Analyze the OCR Text below.
    2. Correct common OCR mistakes based on context (e.g., 'S' -> '5', 'O' -> '0').
    3. **Graceful Handling**: Try your best to logically infer information. If a specific field cannot be found or inferred with confidence, strictly return `null` for that field. Do not makeup data.
    4. Return ONLY the JSON object. Do not include markdown formatting.

    OCR Text:
    '''
    {ocr_text}
    '''
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

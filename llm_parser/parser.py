import os
import json
import ollama
from llm_parser.json_format import Receipt


def run_receipt_llm_parsing(formatted_receipt_text: str, output_filename: str) -> dict:
    """
    Sends the formatted OCR text to Gemma-4 via Ollama, forcing it to output
    data strictly matching the Receipt Pydantic schema, then saves to disk.

    Args:
        formatted_receipt_text: The 2D reconstructed string from parser.py
        output_filename: The base name for the file (e.g., 'receipt2')
    """

    print(f"Sending OCR data to Gemma-4 for {output_filename}...")

    # 1. Inform the LLM of its task with a strong system prompt
    system_prompt = (
        "You are an expert AI extraction assistant. "
        "Your job is to read raw text extracted from a receipt via OCR. "
        "Carefully map the items, prices, totals, and store name into the requested JSON schema. "
        "If a specific field (like tax or subtotal) is missing from the receipt, leave it as null."
    )

    # 2. Call Ollama locally. We use 'format=Receipt.model_json_schema()' to physically
    # constrain the LLM's output tokens so it is incapable of hallucinating invalid JSON.
    response = ollama.chat(
        model='gemma4:e2b',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Extract the receipt data from this text:\n\n{formatted_receipt_text}"}
        ],
        format=Receipt.model_json_schema(),
        options={
            "temperature": 0.0
        }
    )

    # 3. The raw string back from Gemma-4
    raw_llm_json_string = response['message']['content']

    # 4. Validate the string back into our Pydantic object.
    # This guarantees the LLM didn't break our typing rules!
    try:
        parsed_receipt = Receipt.model_validate_json(raw_llm_json_string)
    except Exception as e:
        print(f"Failed to validate LLM output for {output_filename}. Maybe the model hallucinated.")
        print(f"Raw output: {raw_llm_json_string}")
        raise e

    # 5. Save the final robust, validated dictionary to our output folder
    output_dir = "output/json/"
    os.makedirs(output_dir, exist_ok=True)

    final_output_path = os.path.join(output_dir, f"{output_filename}.json")

    # Convert Pydantic object back to dict for easy saving
    final_dict = parsed_receipt.model_dump()

    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=4)

    print(f"Successfully parsed and saved clean JSON to {final_output_path}")

    return final_dict

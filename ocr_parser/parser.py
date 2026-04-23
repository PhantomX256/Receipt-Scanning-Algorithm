import os
import json
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

OUTPUT_JSON_PATH = "output/ocr/"

foundation_predictor = FoundationPredictor()
recognition_predictor = RecognitionPredictor(foundation_predictor)
detection_predictor = DetectionPredictor()


def reconstruct_receipt_text(predictions, y_threshold=15):
    # 1. Strip the noise and only get what we need: (text, and the bounding box)
    lines = []
    for line in predictions["text_lines"]:
        text = line["text"].strip()
        bbox = line["bbox"]  # [min_x, min_y, max_x, max_y]

        if text:  # ignoring empty lines
            lines.append({"text": text, "bbox": bbox})

    # 2. Sort all lines purely by their vertical position (min_y)
    lines.sort(key=lambda x: x["bbox"][1])

    # 3. Group lines into rows if they share roughly the same Y coordinate
    rows = []
    current_row = []
    current_y = None

    for line in lines:
        line_y = line["bbox"][1]

        if current_y is None:
            current_y = line_y
            current_row.append(line)
        elif abs(line_y - current_y) <= y_threshold:
            # It belongs to the same row
            current_row.append(line)
        else:
            # It's a new row! Save the old one and start a new one
            rows.append(current_row)
            current_row = [line]
            current_y = line_y

    if current_row:
        rows.append(current_row)

    # 4. Construct the final string, spacing pieces out based on X-coordinates
    formatted_receipt = ""
    for row in rows:
        # Sort items in this row left-to-right (min_x)
        row.sort(key=lambda x: x["bbox"][0])

        # Join words in the row with a tab so they stay visually separated
        row_text = "\t".join([item["text"] for item in row])
        formatted_receipt += row_text + "\n"

    return formatted_receipt


def run_surya_ocr(processed_receipt, filename):
    pil_receipt = Image.fromarray(processed_receipt)

    predictions = recognition_predictor([pil_receipt], det_predictor=detection_predictor)[0].model_dump()

    os.makedirs(OUTPUT_JSON_PATH, exist_ok=True)
    with open(f"{OUTPUT_JSON_PATH}/{filename}.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    formatted_text = reconstruct_receipt_text(predictions)

    os.makedirs(OUTPUT_JSON_PATH, exist_ok=True)
    with open(f"{OUTPUT_JSON_PATH}/{filename}.txt", "w", encoding="utf-8") as f:
        f.write(formatted_text)

    return formatted_text
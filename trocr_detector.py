# trocr_detector.py

import os
import json
import re
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def trocr_paragraph_from_folder(model_path, lines_folder, output_json_path=None):
    """
    Recognize text line-by-line using a locally downloaded TrOCR model, and return/save a paragraph JSON.
    """
    print(f"🔍 Loading TrOCR model from: {model_path}")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    results = []

    for filename in sorted(os.listdir(lines_folder)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")) and "line_" in filename:
            img_path = os.path.join(lines_folder, filename)
            image = Image.open(img_path).convert("RGB")

            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            results.append({"filename": filename, "text": text})
            print(f"🧾 {filename}: {text}")

    def get_line_number(entry):
        match = re.search(r'line_(\d+)', entry["filename"])
        return int(match.group(1)) if match else 0

    sorted_results = sorted(results, key=get_line_number)
    paragraph_text = " ".join([entry["text"] for entry in sorted_results]).strip()
    paragraph_json = {"paragraph": paragraph_text}

    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(paragraph_json, f, ensure_ascii=False, indent=4)
        print(f"✅ Paragraph JSON saved to: {output_json_path}")

    return paragraph_json

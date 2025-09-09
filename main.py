from craft_detector import detect_text_lines
from trocr_detector import trocr_paragraph_from_folder
import os
import json
import shutil
from output_processing import clean_ocr_text
from llm_structurer import run_llama_and_update_json
from scorer import compute_similarity
from pdf2image import convert_from_path
import numpy as np
import re

# Suppress torchvision warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Function to convert NumPy types to Python types for JSON serialization
def numpy_to_python(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    return obj

def rank_cvs_by_similarity(base_folder):
    """Rank CVs by similarity score from their JSON files."""
    cv_list = []
    
    if not os.path.exists(base_folder):
        print(f"Warning: Base folder {base_folder} does not exist.")
        return cv_list
    
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        file_path = os.path.join(folder_path, "recognized_paragraph.json")
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    similarity = data.get("similarity_score", 0.0)
                    structured_data = data.get("structured_data", {})
                    name = structured_data.get("nom", structured_data.get("name", folder_name))
                    cv_list.append({
                        "folder": folder_name,
                        "nom": name,
                        "score": float(similarity) if similarity is not None else 0.0,
                        "structured_data": structured_data
                    })
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
                cv_list.append({
                    "folder": folder_name,
                    "nom": "Erreur de lecture",
                    "score": 0.0,
                    "structured_data": {}
                })

    ranked_cvs = sorted(cv_list, key=lambda x: x["score"], reverse=True)
    return ranked_cvs

def display_cv_ranking(ranked_cvs):
    """Display the CV ranking with structured data."""
    if not ranked_cvs:
        print("\nAucun CV trouvé pour le classement.")
        return
    
    print("\n📄 Classement des CVs par score de similarité :\n")
    
    for i, cv in enumerate(ranked_cvs, 1):
        score_display = f"{cv['score']:.3f}" if cv['score'] is not None else "N/A"
        print(f"{i}. {cv['nom']} (Dossier: {cv['folder']}, Score: {score_display})")
        
        structured_data = cv.get("structured_data", {})
        if structured_data:
            print("   Structured Data:")
            for key, value in structured_data.items():
                if isinstance(value, list):
                    value_display = ", ".join(str(v) for v in value)
                else:
                    value_display = str(value)
                print(f"     {key}: {value_display}")
        else:
            print("   Structured Data: None")
        print()

# Get absolute path of the current script (main.py)
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))

# Build paths using project_root
data_dir = os.path.join(project_root, 'data')
model_path_craft = os.path.join(project_root, 'weights', 'craft_mlt_25k.pth')
output_dir_Craft = os.path.join(project_root, 'results', 'lines')
model_path_Trocr = os.path.join(project_root, 'weights', 'trocr_base_printed')
temp_dir = os.path.join(project_root, 'temp')

# Specify Poppler path relative to project root
poppler_path = os.path.join(project_root, 'poppler', 'Library', 'bin')

if not os.path.exists(poppler_path):
    print(f"Warning: Poppler path {poppler_path} does not exist. PDF processing may fail.")

os.makedirs(output_dir_Craft, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

def process_pdf(pdf_path, output_dir, poppler_path=None):
    """Convert PDF pages to images and return their paths."""
    image_paths = []
    try:
        print(f"Attempting to convert PDF: {pdf_path}")
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        
        images = convert_from_path(pdf_path, output_folder=temp_dir, poppler_path=poppler_path)
        print(f"Converted {len(images)} pages from {pdf_path}")
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
            image.save(image_path, 'JPEG')
            if os.path.exists(image_path):
                print(f"Saved temporary image: {image_path}")
                image_paths.append(image_path)
            else:
                print(f"Failed to save temporary image: {image_path}")
    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {e}")
        return []
    return image_paths

def process_file(file_path, model_path_craft, model_path_Trocr, output_dir_Craft, jd_text, poppler_path=None):
    """Process a single JPG or PDF file and return JSON path."""
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    lines_folder_Trocr = os.path.join(output_dir_Craft, file_name)
    output_json_Trocr = os.path.join(lines_folder_Trocr, 'recognized_paragraph.json')
    os.makedirs(lines_folder_Trocr, exist_ok=True)

    data = {
        "original_text": "",
        "corrected_text": "",
        "corrected_paragraph": "",
        "structured_data": {},
        "similarity_score": None
    }
    try:
        with open(output_json_Trocr, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Initialized empty JSON at {output_json_Trocr}")
    except Exception as e:
        print(f"Error initializing JSON {output_json_Trocr}: {e}")

    if file_path.lower().endswith('.pdf'):
        image_paths = process_pdf(file_path, output_dir_Craft, poppler_path=poppler_path)
    else:
        image_paths = [file_path]

    if not image_paths:
        print(f"No images generated for {os.path.basename(file_path)}. Skipping further processing.")
        return output_json_Trocr

    corrected_paragraphs = []
    for idx, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f"Image {img_path} does not exist. Skipping.")
            continue

        try:
            boxes, line_images = detect_text_lines(img_path, model_path_craft, result_dir=lines_folder_Trocr)
            print(f"{len(boxes)} lignes détectées pour {os.path.basename(img_path)} dans {lines_folder_Trocr}")
        except Exception as e:
            print(f"Error in CRAFT detection for {os.path.basename(img_path)}: {e}")
            continue

        try:
            trocr_paragraph_from_folder(model_path_Trocr, lines_folder_Trocr, output_json_path=output_json_Trocr)
            print(f"Done with TrOCR Paragraph for {os.path.basename(img_path)}")
        except Exception as e:
            print(f"Error in TrOCR processing for {os.path.basename(img_path)}: {e}")
            continue

        try:
            with open(output_json_Trocr, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"JSON file {output_json_Trocr} not found. Using default data for {os.path.basename(img_path)}.")
            data = {
                "original_text": "",
                "corrected_text": "",
                "corrected_paragraph": "",
                "structured_data": {},
                "similarity_score": None
            }
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON {output_json_Trocr}: {e}")
            data = {
                "original_text": "",
                "corrected_text": "",
                "corrected_paragraph": "",
                "structured_data": {},
                "similarity_score": None
            }

        original_paragraph = data.get("original_text", data.get("paragraph", ""))
        if original_paragraph:
            corrected_paragraph = clean_ocr_text(original_paragraph)
            data[f"corrected_text_page_{idx+1}"] = corrected_paragraph
            corrected_paragraphs.append(corrected_paragraph)
            print(f"Corrected text for page {idx+1}: {corrected_paragraph[:100]}...")
        else:
            print(f"No original text found in JSON for {os.path.basename(img_path)}. Skipping correction.")
            data[f"corrected_text_page_{idx+1}"] = ""

        data = numpy_to_python(data)
        try:
            with open(output_json_Trocr, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Saved corrected text to JSON for {os.path.basename(img_path)}")
        except Exception as e:
            print(f"Error saving JSON for {os.path.basename(img_path)}: {e}")

    combined_corrected_paragraph = " ".join([p for p in corrected_paragraphs if p])
    if corrected_paragraphs:
        data["corrected_text"] = combined_corrected_paragraph
        data["corrected_paragraph"] = combined_corrected_paragraph
        print(f"Combined corrected text: {combined_corrected_paragraph[:100]}...")

        try:
            score = compute_similarity(combined_corrected_paragraph, jd_text)
            if score is not None:
                data["similarity_score"] = float(score)
                print(f"Match Score between CV ({os.path.basename(file_path)}) and JD: {score:.2f}")
            else:
                data["similarity_score"] = None
                print(f"No similarity score computed for {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error computing similarity score for {os.path.basename(file_path)}: {e}")
            data["similarity_score"] = None

        data = numpy_to_python(data)
        try:
            with open(output_json_Trocr, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Saved intermediate JSON with similarity score for {file_name}")
        except Exception as e:
            print(f"Error saving intermediate JSON for {file_name}: {e}")

        try:
            # Stricter truncation to avoid memory issues
            max_text_length = 2500  # Reduced to be safer for CPU
            text_for_llm = combined_corrected_paragraph
            if len(text_for_llm) > max_text_length:
                print(f"⚠️ Text too long ({len(text_for_llm)} chars). Truncating to {max_text_length} chars for LLM processing.")
                text_for_llm = text_for_llm[:max_text_length]
                # Ensure truncation ends at a word boundary
                last_space = text_for_llm.rfind(' ', 0, max_text_length)
                if last_space > 0:
                    text_for_llm = text_for_llm[:last_space]
                data["corrected_paragraph"] = text_for_llm
                # Save truncated text to JSON before LLM processing
                data = numpy_to_python(data)
                with open(output_json_Trocr, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            
            print(f"🚀 Running LLM processing for {os.path.basename(file_path)} (text length: {len(text_for_llm)} chars)...")
            run_llama_and_update_json(output_json_Trocr)
            
            with open(output_json_Trocr, "r", encoding="utf-8") as f:
                updated_data = json.load(f)
            
            data["structured_data"] = updated_data.get("structured_data", {})
            print(f"✅ LLM processing completed for {os.path.basename(file_path)}")
            print(f"Structured data: {json.dumps(data['structured_data'], ensure_ascii=False, indent=4)}")
            
        except Exception as e:
            print(f"❌ Error in LLM phase for {os.path.basename(file_path)}: {e}")
            data["structured_data"] = {}

        data = numpy_to_python(data)
        try:
            with open(output_json_Trocr, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Saved final JSON with all fields for {file_name}")
        except Exception as e:
            print(f"Error saving final JSON for {file_name}: {e}")
    else:
        data = numpy_to_python(data)
        try:
            with open(output_json_Trocr, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Saved empty JSON structure for {file_name}")
        except Exception as e:
            print(f"Error saving empty JSON for {file_name}: {e}")

    return output_json_Trocr

# Main execution
if __name__ == "__main__":
    supported_extensions = ('.jpg', '.jpeg', '.pdf')
    input_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                   if os.path.isfile(os.path.join(data_dir, f)) and f.lower().endswith(supported_extensions)]

    if not input_files:
        print(f"No JPG or PDF files found in {data_dir}")
        exit()

    print("\nPlease enter the job description (JD) text (press Enter twice to finish):")
    jd_lines = []
    while True:
        line = input()
        if line == "":
            break
        jd_lines.append(line)
    jd_text = " ".join(jd_lines)

    processed_files = []
    for file_path in input_files:
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        try:
            output_json_path = process_file(file_path, model_path_craft, model_path_Trocr, output_dir_Craft, jd_text, poppler_path=poppler_path)
            processed_files.append(output_json_path)
            print(f"Done processing {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

    try:
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Error cleaning up temporary directory {temp_dir}: {e}")

    print(f"\nGenerating CV ranking from {output_dir_Craft}...")
    ranked_cvs = rank_cvs_by_similarity(output_dir_Craft)
    display_cv_ranking(ranked_cvs)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
import shutil
import tempfile
from typing import List, Optional
import uuid
from datetime import datetime

# Import your existing modules
from craft_detector import detect_text_lines
from trocr_detector import trocr_paragraph_from_folder
from output_processing import clean_ocr_text
from llm_structurer import run_llama_and_update_json
from scorer import compute_similarity
from pdf2image import convert_from_path
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

app = FastAPI(
    title="CV Processing API",
    description="API for processing CVs against job descriptions and ranking by similarity",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration paths - matching your original structure
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))  # Go up one level from src/

# Build paths using project_root (same as your original code)
MODEL_PATH_CRAFT = os.path.join(project_root, 'weights', 'craft_mlt_25k.pth')
MODEL_PATH_TROCR = os.path.join(project_root, 'weights', 'trocr_base_printed')
POPPLER_PATH = os.path.join(project_root, 'poppler', 'Library', 'bin')
UPLOAD_DIR = os.path.join(project_root, 'uploads')
RESULTS_DIR = os.path.join(project_root, 'results', 'lines')  # Match your original output_dir_Craft
TEMP_DIR = os.path.join(project_root, 'temp')

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def numpy_to_python(obj):
    """Convert NumPy types to Python types for JSON serialization."""
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

def process_pdf(pdf_path, temp_dir, poppler_path=None):
    """Convert PDF pages to images and return their paths."""
    image_paths = []
    try:
        print(f"Converting PDF: {pdf_path}")
        images = convert_from_path(pdf_path, output_folder=temp_dir, poppler_path=poppler_path)
        print(f"Converted {len(images)} pages")
        
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
            image.save(image_path, 'JPEG')
            image_paths.append(image_path)
            
    except Exception as e:
        print(f"Error converting PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")
    
    return image_paths

def process_single_cv(file_path, jd_text, session_id):
    """Process a single CV file and return the results."""
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    lines_folder = os.path.join(RESULTS_DIR, file_name)  # Use RESULTS_DIR directly like original
    output_json = os.path.join(lines_folder, 'recognized_paragraph.json')
    
    os.makedirs(lines_folder, exist_ok=True)
    
    # Initialize data structure
    data = {
        "original_text": "",
        "corrected_text": "",
        "corrected_paragraph": "",
        "structured_data": {},
        "similarity_score": None,
        "file_name": file_name,
        "processing_timestamp": datetime.now().isoformat()
    }
    
    # Save initial structure
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    # Handle PDF or image files
    if file_path.lower().endswith('.pdf'):
        session_temp_dir = os.path.join(TEMP_DIR, session_id)
        os.makedirs(session_temp_dir, exist_ok=True)
        image_paths = process_pdf(file_path, session_temp_dir, POPPLER_PATH)
    else:
        image_paths = [file_path]
    
    if not image_paths:
        raise HTTPException(status_code=400, detail="No images could be processed from the file")
    
    corrected_paragraphs = []
    
    for idx, img_path in enumerate(image_paths):
        try:
            # CRAFT text detection
            boxes, line_images = detect_text_lines(img_path, MODEL_PATH_CRAFT, result_dir=lines_folder)
            print(f"Detected {len(boxes)} text lines for page {idx+1}")
            
            # TrOCR text recognition
            trocr_paragraph_from_folder(MODEL_PATH_TROCR, lines_folder, output_json_path=output_json)
            
            # Load and process results
            with open(output_json, "r", encoding="utf-8") as f:
                page_data = json.load(f)
            
            original_text = page_data.get("original_text", page_data.get("paragraph", ""))
            if original_text:
                corrected_text = clean_ocr_text(original_text)
                data[f"corrected_text_page_{idx+1}"] = corrected_text
                corrected_paragraphs.append(corrected_text)
            
        except Exception as e:
            print(f"Error processing page {idx+1}: {e}")
            continue
    
    # Combine all corrected text
    combined_text = " ".join([p for p in corrected_paragraphs if p])
    if combined_text:
        data["corrected_text"] = combined_text
        data["corrected_paragraph"] = combined_text
        
        # Compute similarity score
        try:
            score = compute_similarity(combined_text, jd_text)
            data["similarity_score"] = float(score) if score is not None else None
        except Exception as e:
            print(f"Error computing similarity: {e}")
            data["similarity_score"] = None
        
        # LLM structuring (with text truncation for safety)
        try:
            max_text_length = 2500
            text_for_llm = combined_text
            if len(text_for_llm) > max_text_length:
                text_for_llm = text_for_llm[:max_text_length]
                last_space = text_for_llm.rfind(' ', 0, max_text_length)
                if last_space > 0:
                    text_for_llm = text_for_llm[:last_space]
                data["corrected_paragraph"] = text_for_llm
            
            # Save before LLM processing
            data = numpy_to_python(data)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            # Run LLM structuring
            run_llama_and_update_json(output_json)
            
            # Load structured data
            with open(output_json, "r", encoding="utf-8") as f:
                updated_data = json.load(f)
            data["structured_data"] = updated_data.get("structured_data", {})
            
        except Exception as e:
            print(f"Error in LLM processing: {e}")
            data["structured_data"] = {}
    
    # Final save
    data = numpy_to_python(data)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return data

def cleanup_session_files(session_id):
    """Clean up temporary files for a session."""
    try:
        session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
        session_temp_dir = os.path.join(TEMP_DIR, session_id)
        
        if os.path.exists(session_upload_dir):
            shutil.rmtree(session_upload_dir)
        if os.path.exists(session_temp_dir):
            shutil.rmtree(session_temp_dir)
            
    except Exception as e:
        print(f"Error cleaning up session {session_id}: {e}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "CV Processing API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check."""
    health_status = {
        "status": "healthy",
        "models": {
            "craft_model_exists": os.path.exists(MODEL_PATH_CRAFT),
            "trocr_model_exists": os.path.exists(MODEL_PATH_TROCR)
        },
        "directories": {
            "upload_dir": os.path.exists(UPLOAD_DIR),
            "results_dir": os.path.exists(RESULTS_DIR),
            "temp_dir": os.path.exists(TEMP_DIR)
        }
    }
    return health_status

@app.post("/process-cvs")
async def process_cvs(
    job_description: str = Form(..., description="Job description text"),
    files: List[UploadFile] = File(..., description="CV files (JPG, JPEG, PDF)")
):
    """
    Process multiple CV files against a job description.
    Returns ranked results by similarity score.
    """
    session_id = str(uuid.uuid4())
    session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
    session_results_dir = os.path.join(RESULTS_DIR, session_id)
    
    os.makedirs(session_upload_dir, exist_ok=True)
    os.makedirs(session_results_dir, exist_ok=True)
    
    try:
        # Validate files
        supported_extensions = ('.jpg', '.jpeg', '.pdf')
        valid_files = []
        
        for file in files:
            if not file.filename.lower().endswith(supported_extensions):
                continue
            valid_files.append(file)
        
        if not valid_files:
            raise HTTPException(
                status_code=400, 
                detail="No valid files found. Supported formats: JPG, JPEG, PDF"
            )
        
        # Save uploaded files
        saved_files = []
        for file in valid_files:
            file_path = os.path.join(session_upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_files.append(file_path)
        
        # Process each CV
        results = []
        for file_path in saved_files:
            try:
                print(f"Processing: {os.path.basename(file_path)}")
                cv_data = process_single_cv(file_path, job_description, session_id)
                
                # Extract key information for response
                result = {
                    "file_name": cv_data.get("file_name", os.path.basename(file_path)),
                    "similarity_score": cv_data.get("similarity_score"),
                    "structured_data": cv_data.get("structured_data", {}),
                    "corrected_text_preview": cv_data.get("corrected_text", "")[:200] + "..." if cv_data.get("corrected_text", "") else ""
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {os.path.basename(file_path)}: {e}")
                results.append({
                    "file_name": os.path.basename(file_path),
                    "similarity_score": None,
                    "structured_data": {},
                    "error": str(e),
                    "corrected_text_preview": ""
                })
        
        # Rank by similarity score
        ranked_results = sorted(
            results, 
            key=lambda x: x.get("similarity_score") or 0, 
            reverse=True
        )
        
        # Clean up uploaded files
        cleanup_session_files(session_id)
        
        return {
            "session_id": session_id,
            "job_description": job_description,
            "total_files_processed": len(results),
            "ranked_cvs": ranked_results,
            "processing_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        # Clean up on error
        cleanup_session_files(session_id)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process-single-cv")
async def process_single_cv_endpoint(
    job_description: str = Form(..., description="Job description text"),
    file: UploadFile = File(..., description="Single CV file (JPG, JPEG, PDF)")
):
    """
    Process a single CV file against a job description.
    """
    session_id = str(uuid.uuid4())
    session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
    
    os.makedirs(session_upload_dir, exist_ok=True)
    
    try:
        # Validate file
        supported_extensions = ('.jpg', '.jpeg', '.pdf')
        if not file.filename.lower().endswith(supported_extensions):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Supported formats: JPG, JPEG, PDF"
            )
        
        # Save uploaded file
        file_path = os.path.join(session_upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process CV
        cv_data = process_single_cv(file_path, job_description, session_id)
        
        # Clean up
        cleanup_session_files(session_id)
        
        return {
            "session_id": session_id,
            "file_name": cv_data.get("file_name", file.filename),
            "similarity_score": cv_data.get("similarity_score"),
            "structured_data": cv_data.get("structured_data", {}),
            "corrected_text": cv_data.get("corrected_text", ""),
            "processing_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        cleanup_session_files(session_id)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/results/{session_id}")
async def get_session_results(session_id: str):
    """
    Retrieve detailed results for a specific session.
    """
    # Look directly in RESULTS_DIR since we're not using session subdirectories
    if not os.path.exists(RESULTS_DIR):
        raise HTTPException(status_code=404, detail="Results directory not found")
    
    try:
        results = []
        for folder_name in os.listdir(RESULTS_DIR):
            folder_path = os.path.join(RESULTS_DIR, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            json_path = os.path.join(folder_path, "recognized_paragraph.json")
            if os.path.isfile(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append(data)
        
        # Rank by similarity score
        ranked_results = sorted(
            results,
            key=lambda x: x.get("similarity_score") or 0,
            reverse=True
        )
        
        return {
            "session_id": session_id,
            "total_cvs": len(ranked_results),
            "results": ranked_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")

@app.delete("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """
    Clean up all files for a specific session.
    """
    try:
        session_results_dir = os.path.join(RESULTS_DIR, session_id)
        if os.path.exists(session_results_dir):
            shutil.rmtree(session_results_dir)
        
        cleanup_session_files(session_id)
        
        return {"message": f"Session {session_id} cleaned up successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """
    List all available session results.
    """
    try:
        sessions = []
        if os.path.exists(RESULTS_DIR):
            for session_id in os.listdir(RESULTS_DIR):
                session_path = os.path.join(RESULTS_DIR, session_id)
                if os.path.isdir(session_path):
                    cv_count = len([d for d in os.listdir(session_path) 
                                  if os.path.isdir(os.path.join(session_path, d))])
                    sessions.append({
                        "session_id": session_id,
                        "cv_count": cv_count,
                        "created": datetime.fromtimestamp(
                            os.path.getctime(session_path)
                        ).isoformat()
                    })
        
        return {"sessions": sessions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")

if __name__ == "__main__":
    # Check if required models exist
    if not os.path.exists(MODEL_PATH_CRAFT):
        print(f"Warning: CRAFT model not found at {MODEL_PATH_CRAFT}")
    if not os.path.exists(MODEL_PATH_TROCR):
        print(f"Warning: TrOCR model not found at {MODEL_PATH_TROCR}")
    if not os.path.exists(POPPLER_PATH):
        print(f"Warning: Poppler not found at {POPPLER_PATH}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
import subprocess
import json
import os
import unicodedata

# Paths - adjust as needed
LLAMA_RUN_PATH = r"C:\Users\USER\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-run.exe"
MODEL_PATH = r"C:\llama\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"

def create_prompt(cv_text: str) -> str:
    """Create the LLaMA prompt to extract structured JSON from CV text."""
    # Normalize unicode to ASCII to avoid encoding issues
    cv_text = unicodedata.normalize('NFKD', cv_text).encode('ascii', 'ignore').decode('ascii')
    # Escape curly braces to avoid formatting issues if you use .format() later
    if '{' in cv_text or '}' in cv_text:
        cv_text = cv_text.replace('{', '{{').replace('}', '}}')
    prompt = (
        "You are an expert in human resources. Below is the full text extracted from a CV, "
        "which may contain typos, mixed languages, or irregular formatting:\n"
        f"{cv_text}\n\n"
        "Extract and structure the following information into a JSON object:\n"
        '- "nom": Full name of the candidate\n'
        '- "poste": Most recent job title\n'
        '- "email": Email address\n'
        '- "experience": List of work experiences, each with:\n'
        '    - "poste" (job title),\n'
        '    - "entreprise" (company name,Society name , do not include emails or personal names),\n'
        '    - "date_debut" (start date, e.g., "Juin 2019"),\n'
        '    - "date_fin" (end date, e.g., "Janvier 2020" or "Present" if ongoing)\n'
        '- "competences": List of skills\n'
        '- "formation": List of education, each with "diplome" (degree), "etablissement" (institution), "annee" (year, e.g., "2018")\n\n'
        "Return only the JSON object, enclosed in triple backticks as follows:\n"
        "```json\n"
        '{\n'
        '  "key": "value"\n'
        '}\n'
        "```\n"
        "Ignore typos (e.g., 'DUILLET' for 'JUILLET') and extract the most accurate information possible. "
        "Ensure the JSON is valid and properly formatted."
    )
    return prompt

def run_llama(cv_text: str, gpu_layers: int = 10, timeout: int = 600):
    """Run the llama.cpp executable with the prompt to get structured CV data."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return None

    if not os.path.exists(LLAMA_RUN_PATH):
        print(f"❌ LLaMA executable not found: {LLAMA_RUN_PATH}")
        print("Please ensure llama.cpp is installed correctly via Winget or compiled with Vulkan support.")
        return None

    prompt = create_prompt(cv_text)
    
    try:
        print(f"🔄 Running LLaMA (GPU layers: {gpu_layers})...")
        print(f"📁 Model: {MODEL_PATH}")
        print(f"📝 Prompt length: {len(prompt)} characters")
        print("-" * 50)

        process = subprocess.Popen(
            [
                LLAMA_RUN_PATH,
                MODEL_PATH,
                "--context-size", "4096",
                "--temp", "0.1",
                "--ngl", str(gpu_layers),
                "--threads", "6",
                "--verbose"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        stdout, stderr = process.communicate(input=prompt, timeout=timeout)

        if process.returncode != 0:
            print(f"❌ Error running llama.cpp:")
            print(f"Return code: {process.returncode}")
            print(f"STDERR: {stderr}")
            if "memory" in stderr.lower() or "vulkan" in stderr.lower():
                print("⚠️ Possible GPU memory issue. Retrying with CPU only...")
                return run_llama(cv_text, gpu_layers=0, timeout=timeout)
            return None

        print("✅ LLaMA execution completed!")
        print("=== RAW OUTPUT ===")
        print(stdout)
        print("=== END RAW OUTPUT ===")
        print("=== STDERR ===")
        print(stderr)
        print("=== END STDERR ===")

        # Extract JSON between ```json and ```
        start_marker = "```json\n"
        end_marker = "\n```"
        start_idx = stdout.find(start_marker)
        end_idx = stdout.rfind(end_marker)

        if start_idx != -1 and end_idx > start_idx:
            json_str = stdout[start_idx + len(start_marker):end_idx].strip()
            try:
                parsed_json = json.loads(json_str)
                print("✅ Parsed JSON:")
                print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"⚠️ Could not parse JSON: {e}")
                print("Raw JSON string:", json_str)
                return stdout
        else:
            print("⚠️ No JSON block found in output")
            return stdout

    except subprocess.TimeoutExpired:
        process.kill()
        print(f"❌ LLaMA execution timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def run_llama_and_update_json(json_path: str, gpu_layers: int = 10, timeout: int = 600):
    """
    Runs LLaMA on the 'corrected_paragraph' from a JSON file and saves the result under 'structured_data'.
    """
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corrected_text = data.get("corrected_paragraph", "")
    if not corrected_text:
        print("❌ 'corrected_paragraph' not found in JSON.")
        return None

    print("🚀 Starting LLaMA on corrected paragraph...")
    result = run_llama(corrected_text, gpu_layers=gpu_layers, timeout=timeout)

    if isinstance(result, dict):
        # CORRECTION: Changer "structured" en "structured_data"
        data["structured_data"] = result
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"✅ Structured data saved to {json_path}")
        return result
    else:
        print("⚠️ LLaMA did not return structured JSON.")
        return result


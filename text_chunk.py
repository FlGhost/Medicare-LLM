import os
import json
import time
from pathlib import Path
from google import genai

# --- !! IMPORTANT !! ---
# SET YOUR API KEY IN YOUR ENVIRONMENT
# On Windows (Command Prompt): set GEMINI_API_KEY=YOUR_API_KEY_HERE
# On Windows (PowerShell):   $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
# On macOS/Linux:           export GEMINI_API_KEY='YOUR_API_KEY_HERE'
# -------------------------
try:
    API_KEY = os.environ["GEMINI_API_KEY"]
except KeyError:
    print("="*80)
    print("!! ERROR: GEMINI_API_KEY environment variable not set. !!")
    print("Please set the environment variable before running this script.")
    print("="*80)
    exit()

# Use Pathlib for cross-platform compatibility
INPUT_DIR = Path(r"D:\Medicare\Medicare-LLM\text_output")
OUTPUT_DIR = Path(r"D:\Medicare\Medicare-LLM\text_output_chunked")

MODEL_NAME = "gemini-2.5-pro" 

# delay in seconds between API calls to avoid rate limiting, 2.5 has 2 per minute so
REQUEST_DELAY_SEC = 31 


SYSTEM_PROMPT = """## System
You are **TropID-Extractor**, an expert clinical information extractor for **tropical & infectious diseases**.  
Your task: **read a free-text clinical case in JSON format** and return a **single JSON object** where **each section is one coherent full-text block** (no bulletizing into tiny subfields).  
If a section is not present, set it to `null`. **Do not guess.** If there are irrelevant info, they can be left out.
**Output ONLY valid JSON** — no preamble, no commentary.
### Formatting & Safety Rules
- **One JSON object only.**
- **Full-text blocks:** Each field below must be a **cohesive paragraph** (or short multi-sentence block) stitched from the case text; paraphrase minimally, preserve clinical meaning, and **do not invent** missing details.
- **Attribution discipline:** Prefer exact phrases from the source for key facts (fever pattern, exposures, test names, titers) but keep the prose readable.
- **Units & names:** Keep units and proper names as written (°C/°F, NS1, thick smear, RDT, species, titers/CT values).
- **Privacy:** Exclude any direct identifiers if present.
- **Uncertainty:** If the case explicitly says something is “unclear/unknown,” include that wording.
- **Final diagnosis:** Write a concise paragraph that **states the diagnosis, the causative agent if given, and the evidence** (labs/imaging/epidemiology/response to therapy).  
- **Disease name (short):** After `final_diagnosis`, fill `disease_name_short` with the **best disease name only** (e.g., “Dengue fever”, “Falciparum malaria”, “Scrub typhus”).
---
Return **only** this JSON schema (exact keys, same order):
```json
{
  "patient_information": null,
  "chief_complaint": null,
  "history_of_present_illness": null,
  "exposure_and_epidemiology": null,
  "vitals": null,
  "physical_exam": null,
  "labs_and_diagnostics": null,
  "differential_diagnosis": null,
  "management_and_clinical_course": null,
  "final_diagnosis": null,
  "disease_name_short": null
}
```

### Field guidance (concise)
- **patient_information**: age/sex; relevant comorbidities/immunosuppression; vaccination/allergy info if stated.  
- **chief_complaint**: one-line problem + duration.  
- **history_of_present_illness**: timeline, key symptoms, pertinent negatives, severity pattern.  
- **exposure_and_epidemiology**: residence/travel (place/setting, dates if present), vectors (mosquito/tick), animals, water/food risks, contacts, season/outbreak context, occupation.  
- **vitals**: all reported vital signs as text (fever values, BP, HR, RR, SpO₂).  
- **physical_exam**: salient systems (skin, HEENT, chest, abdo, neuro, lymph, etc.).  
- **labs_and_diagnostics**: CBC trends, key chem/coag, inflammatory markers, microbiology/serology/PCR (assay + result + titer/CT if present), malaria tests, imaging highlights.  
- **differential_diagnosis**: succinct narrative of the main alternatives considered, with one-sentence justification for/against each.
- **management_and_clinical_course**: antimicrobials (drug/dose if given), supportive care, procedures; response, complications, outcome.  
- **final_diagnosis**: 3–5 sentences: explicit disease name ± causative agent, confirmation method (e.g., NS1+, thick smear species, PCR/serology), and why alternatives were ruled out.  
- **disease_name_short**: the disease name only (no agent, no method).  
"""


def process_single_file(client, input_path, output_path):
    """
    Reads one JSON file, sends its content to the Gemini API, 
    and saves the structured JSON response.
    """
    try:
        # read the source JSON file
        with open(input_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        # convert the source JSON object back into a string to feed to the model
        case_data_string = json.dumps(source_data, indent=2)
        # combine SYSTEM_PROMPT with case data
        full_query = f"{SYSTEM_PROMPT}\n\n---\n\nExtract information from the following clinical case:\n{case_data_string}"

    except json.JSONDecodeError:
        print(f"   ... ERROR: Failed to decode source JSON. Skipping.")
        return
    except Exception as e:
        print(f"   ... ERROR reading file: {e}. Skipping.")
        return

    try:
        config_dict = {
            "response_mime_type": "application/json"
        }
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[full_query],
            config=config_dict 
        )

        # The API is forced to return JSON, but we clean it
        # just in case it includes markdown backticks
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "")

        try:
            output_json = json.loads(cleaned_response_text)
        except json.JSONDecodeError:
            print(f"   ... ERROR: API did not return valid JSON. Response was:")
            print(cleaned_response_text)
            return

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=2)
        
        print(f"   ... Success! Saved to {output_path.name}")

    except Exception as e:
        print(f"   ... ERROR during API call: {e}")


def main():
    """
    Main function to orchestrate the batch processing.
    """
    print("--- Tropical Disease Case Extractor ---")
    print(f"Input folder:  {INPUT_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Model:         {MODEL_NAME}")
    print("-" * 40)

    OUTPUT_DIR.mkdir(exist_ok=True)
    client = genai.Client(api_key=API_KEY)


    # Get all .json files from the input directory
    input_files = list(INPUT_DIR.glob("*.json"))
    
    if not input_files:
        print(f"No .json files found in {INPUT_DIR}. Exiting.")
        return

    total_files = len(input_files)
    print(f"Found {total_files} JSON files to process.")

    for i, input_path in enumerate(input_files):
        print(f"\n[{i+1}/{total_files}] Processing: {input_path.name}")
        output_path = OUTPUT_DIR / input_path.name

        # check if output already exists ---
        if output_path.exists():
            print("   ... Output file already exists. Skipping.")
            continue
        
        # process the file
        process_single_file(client, input_path, output_path)
        
        # rate limit
        if i < total_files - 1:
            time.sleep(REQUEST_DELAY_SEC)

    print("\n" + "-" * 40)
    print("Batch processing complete.")

if __name__ == "__main__":
    main()

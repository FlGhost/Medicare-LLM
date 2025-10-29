import re
from pathlib import Path

OUTPUT_DIR = Path(r"D:\Medicare\Medicare-LLM\text_output_chunked")
#this exist because the gemini API output fucks up the unicodes
UNICODE_REPLACEMENTS = {
    r"\u00b0": "°",    # Degree Sign
    r"\u00b2": "²",    # Superscript Two
    r"\u00b5": "µ",    # Micro Sign
    r"\u00ef": "ï",    # Latin Small Letter I with Diaeresis
    r"\u2013": "–",    # En-dash
    r"\u2014": "—",    # Em-dash
    r"\u2019": "’",    # Right Single Quote (Apostrophe)
    r"\u201c": "“",    # Left Double Quotation Mark
    r"\u201d": "”",    # Right Double Quotation Mark
    r"\u2026": "…",    # Ellipsis
    r"\u2079": "⁹",    # Superscript Nine
    r"\u03bc": "μ",    # Greek Small Letter Mu
    r"\u03b2": "β",    # Greek Small Letter Beta
}

def fix_unicode_in_files(target_dir):
    """
    Iterates through all JSON files in the target directory, reads them as 
    raw text, performs string replacement for specific escaped Unicode characters, 
    and writes the fixed content back.
    """
    print(f"--- Starting Unicode Fix for JSON files in: {target_dir} ---")
    
    if not target_dir.exists():
        print(f"Error: Directory not found at {target_dir}")
        return

    json_files = list(target_dir.glob("*.json"))
    
    if not json_files:
        print("No JSON files found. Exiting.")
        return
        
    total_files = len(json_files)
    print(f"Found {total_files} JSON files to process.")

    for i, file_path in enumerate(json_files):
        print(f"[{i+1}/{total_files}] Processing: {file_path.name}")
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            modified_content = content
            for escaped_str, replacement_char in UNICODE_REPLACEMENTS.items():
                modified_content = modified_content.replace(escaped_str, replacement_char)
            
            file_path.write_text(modified_content, encoding='utf-8')
            
            print("   ... Fixed and saved.")
            
        except Exception as e:
            print(f"   ... ERROR processing {file_path.name}: {e}")

    print("\n--- Unicode fixing complete. ---")

if __name__ == "__main__":
    fix_unicode_in_files(OUTPUT_DIR)

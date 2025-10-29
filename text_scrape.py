import fitz
import os
import json
import re

data_folder = "D:\\Medicare\\Medicare-LLM\\data"
output_folder = "D:\\Medicare\\Medicare-LLM\\text_output" 
# create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

all_pdf_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]

print(f"Found {len(all_pdf_files)} PDF files to process.")

def cleanup_text(text):
    # regex shtuffsdjk
    text = re.sub(r'\n+', '\n', text)
    text = '\n'.join(line.strip() for line in text.splitlines())
    text = re.sub(r'^\d+$|^\w$', '', text, flags=re.MULTILINE)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'-\s+', '', text)
    return text.strip()

for pdf_path in all_pdf_files:
    try:
        print(f"Attempting to open: {pdf_path}") 
        doc = fitz.open(pdf_path)

        if doc.is_encrypted:
            print(f"  PDF '{os.path.basename(pdf_path)}' is encrypted. Skipping.")
            doc.close()
            continue

        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        output_filename = f"{base_filename}.json"
        output_path = os.path.join(output_folder, output_filename)

        pdf_data = {
            "filename": os.path.basename(pdf_path),
            "pages": []
        }

        print(f"Extracting text from: {os.path.basename(pdf_path)}")
        for i, page in enumerate(doc): # page number for debugging
            try:
                text = page.get_text()
                cleaned_text = cleanup_text(text)
                pdf_data["pages"].append({
                    "page_number": i + 1,
                    "text": cleaned_text
                })
            except Exception as page_e:
                print(f"  Error extracting text from page {i+1} in {os.path.basename(pdf_path)}: {page_e}")
                pdf_data["pages"].append({
                    "page_number": i + 1,
                    "text": f"Error extracting text: {str(page_e)}"
                })
                continue # skip[ if bad

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pdf_data, f, ensure_ascii=False, indent=4)

        doc.close()
        print(f"  Saved text from {os.path.basename(pdf_path)} to {output_filename}")

    except Exception as e:
        print(f"Error processing {os.path.basename(pdf_path)}: {e}")

print("\nText extraction complete.")
import os
import fitz  # PyMuPDF
from PIL import Image
import torch
from typing import List, Dict, Any

# Import the custom classes for the ColQwen2.5 model
try:
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    COLPALI_ENGINE_AVAILABLE = True
except ImportError:
    COLPALI_ENGINE_AVAILABLE = False


class ColPaliRetriever:
    """
    A class to perform visual document retrieval on a PDF.
    It indexes all pages of a PDF and allows searching with a text query.
    """

    def __init__(self, model_id: str):
        if not COLPALI_ENGINE_AVAILABLE:
            raise RuntimeError(
                "colpali-engine is not available. Please install it using 'pip install colpali-engine'"
            )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load the model and processor using the custom classes
        try:
            self.model = ColQwen2_5.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
                device_map=self.device,
            ).eval()
            self.processor = ColQwen2_5_Processor.from_pretrained(model_id)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model '{model_id}': {e}")

        self.page_index: List[Dict[str, Any]] = []

    def _render_page(self, page: fitz.Page) -> Image.Image:
        """Renders a fitz.Page object to a PIL Image."""
        mat = fitz.Matrix(2, 2)  # Render at 144 DPI (72 * 2)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def index_pdf(self, pdf_path: str):
        """
        Processes every page of a PDF and stores its visual embedding in an in-memory index.
        """
        print(f"\n--- Starting Indexing of '{os.path.basename(pdf_path)}' ---")
        self.page_index = []
        doc = fitz.open(pdf_path)

        for i, page in enumerate(doc):
            print(f"Indexing page {i + 1}/{len(doc)}...")
            image = self._render_page(page)
            
            with torch.no_grad():
                batch_images = self.processor.process_images([image]).to(self.device)
                image_embedding = self.model(**batch_images)

            self.page_index.append({
                "page_number": i + 1,
                "embedding": image_embedding.cpu() # Store on CPU to save GPU memory
            })
        
        print("--- Indexing Complete ---")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the indexed pages for the most relevant matches to the text query.
        """
        if not self.page_index:
            raise ValueError("The PDF has not been indexed yet. Please call .index_pdf() first.")

        print(f"\n--- Searching for query: '{query}' ---")
        
        # 1. Create the query embedding
        with torch.no_grad():
            batch_queries = self.processor.process_queries([query]).to(self.device)
            query_embedding = self.model(**batch_queries)

        # 2. Score the query against every page in the index
        results = []
        for item in self.page_index:
            image_embedding = item["embedding"].to(self.device) # Move back to GPU for scoring
            score = self.processor.score_multi_vector(query_embedding, image_embedding).item()
            results.append({
                "page_number": item["page_number"],
                "score": score
            })

        # 3. Rank the results and return the top_k
        ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return ranked_results[:top_k]

# ---------------------------
# Example: Document Retrieval Workflow
# ---------------------------
if __name__ == "__main__":
    pdf_file = r"D:\Personal tasks\Medicare-LLM\data\1---A-20-Year-Old-Woman-from-Sudan-With-Fever--_2022_Clinical-Cases-in-Tropi.pdf"
    
    # Use the correct model ID for the visual retriever
    model_id = "vidore/colqwen2.5-v0.2"

    try:
        # Initialize the retriever
        retriever = ColPaliRetriever(model_id=model_id)

        # STEP 1: Index the entire PDF document. This can take a moment.
        retriever.index_pdf(pdf_path=pdf_file)

        # STEP 2: Define a search query and search the index.
        search_query = "A patient from Sudan with a high temperature"
        top_results = retriever.search(query=search_query, top_k=3)

        # STEP 3: Display the ranked results.
        print("\n--- Top 3 Matching Pages ---")
        for result in top_results:
            print(f"Page: {result['page_number']} (Score: {result['score']:.4f})")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
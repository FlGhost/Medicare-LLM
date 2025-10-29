import google.generativeai as genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from python_scripts.config import (
    QDRANT_ENDPOINT_URL, 
    QDRANT_API_KEY, 
    HUGGINGFACETOKEN, 
    GEMINI_API_KEY, 
    logger
)
from typing import List, Dict, Any

# --- Configuration ---
# This is YOUR chosen embedding model
EMBEDDING_MODEL_NAME = 'pritamdeka/S-BioBert-snli-multinli-stsb'

# This is your generative model (the "chat agent")
GENERATOR_MODEL_NAME = 'gemini-2.5-flash'

# This is the Qdrant collection you uploaded your data to
COLLECTION_NAME = "main-rag"

class RAGAgent:
    def __init__(self):
        """
        Initializes the RAG agent by loading all necessary models and clients.
        """
        logger.info("Initializing RAGAgent...")
        
        # 1. Initialize Embedding Model (for querying)
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embed_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, 
            use_auth_token=HUGGINGFACETOKEN
        )
        logger.info("Embedding model loaded.")
        
        # 2. Initialize Qdrant Client (for retrieving)
        logger.info(f"Connecting to Qdrant Cloud...")
        if not QDRANT_ENDPOINT_URL or not QDRANT_API_KEY:
            raise ValueError("QDRANT_ENDPOINT_URL or QDRANT_API_KEY not found in .env file.")
        
        self.qdrant_client = QdrantClient(
            url=QDRANT_ENDPOINT_URL, 
            api_key=QDRANT_API_KEY
        )
        logger.info("Qdrant Cloud connection successful.")
        
        # 3. Initialize Generative Model (for generating)
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
            
        logger.info(f"Initializing generative model: {GENERATOR_MODEL_NAME}")
        genai.configure(api_key=GEMINI_API_KEY)
        self.gen_model = genai.GenerativeModel(GENERATOR_MODEL_NAME)
        
        logger.info("RAGAgent initialized successfully.")

    def search_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Embeds the query and searches Qdrant for the top_k most relevant documents.
        """
        logger.info(f"Embedding query: '{query[:50]}...'")
        # Use your S-BioBert model to create the query vector
        query_vector = self.embed_model.encode(query).tolist()
        
        logger.info(f"Searching collection '{COLLECTION_NAME}' in Qdrant...")
        search_results = self.qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True  # This is crucial! It returns your JSON data
        )
        
        # Extract just the payloads (your original JSON) from the search results
        retrieved_contexts = [result.payload for result in search_results]
        logger.info(f"Retrieved {len(retrieved_contexts)} contexts.")
        return retrieved_contexts

    def build_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Builds a comprehensive prompt for the generative model,
        using the structured JSON context you created.
        """
        # Convert the list of JSON payloads into a readable context string
        context_str = "\n\n---\n\n".join(
            [f"Source Document: {doc.get('source_filename', 'N/A')}\n"
             f"Diagnosis: {doc.get('final_diagnosis', 'N/A')}\n"
             f"Disease Name: {doc.get('disease_name_short', 'N/A')}\n"
             f"History: {doc.get('history_of_present_illness', 'N/A')}\n"
             f"Labs: {doc.get('labs_and_diagnostics', 'N/A')}" 
             for doc in context_docs]
        )
        
        prompt = f"""
        You are an expert clinical diagnostic assistant. Your task is to answer the user's question based *only* on the provided clinical case summaries.
        
        Do not use any external knowledge. If the answer is not in the provided context, state that clearly: "I could not find an answer in the provided case reports."
        
        **PROVIDED CONTEXT:**
        {context_str}
        
        **USER QUESTION:**
        {query}
        
        **ASSISTANT ANSWER:**
        """
        return prompt

    def ask(self, query: str) -> str:
        """
        Main method to run the full RAG pipeline.
        Query -> Embed -> Search -> Augment -> Generate
        """
        # 1. Retrieve context
        retrieved_contexts = self.search_knowledge_base(query)
        
        if not retrieved_contexts:
            logger.warning("No relevant context found in the database.")
            return "I'm sorry, I could not find any relevant information in the clinical cases to answer your question."
            
        # 2. Build the prompt
        prompt = self.build_prompt(query, retrieved_contexts)
        
        # 3. Generate the answer
        logger.info("Generating final answer from context...")
        try:
            # Set safety settings to be less restrictive (medical data can be sensitive)
            safety_settings = {
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
            response = self.gen_model.generate_content(prompt, safety_settings=safety_settings)
            return response.text
        except Exception as e:
            logger.error(f"Error during answer generation: {e}")
            return f"An error occurred while generating the response: {e}"

# --- Main execution block to test the agent ---
if __name__ == "__main__":
    """
    This part lets you test the agent directly by running:
    python python_scripts/rag_agent.py
    """
    try:
        agent = RAGAgent()
        
        # --- Test Query ---
        test_query = "What are the common symptoms and lab results for Dengue?"
        
        print(f"\nTesting agent with query: '{test_query}'\n")
        answer = agent.ask(test_query)
        
        print("\n--- GENERATED ANSWER ---")
        print(answer)
        print("--------------------------")
        
    except Exception as e:
        logger.error(f"Failed to run RAGAgent: {e}")
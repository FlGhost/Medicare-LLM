import os
import json
import uuid
from pathlib import Path
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
# Import the new Qdrant config variables
from python_scripts.config import HUGGINGFACETOKEN, QDRANT_ENDPOINT_URL, QDRANT_API_KEY, logger 
from typing import Dict, Any, List

# --- Configuration ---
INPUT_DIR = Path(r"E:\Medicare-LLM\text_output_chunked")
COLLECTION_NAME = "main-rag" # Using your cluster name

EMBEDDING_MODEL_NAME = 'pritamdeka/S-BioBert-snli-multinli-stsb'
EMBEDDING_DIMENSION = 768 

def get_embedding_model(model_name: str, token: str) -> SentenceTransformer:
    """
    Initializes and returns the SentenceTransformer model.
    """
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name, use_auth_token=token)
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        logger.error("Make sure your HUGGINGFACETOKEN is set correctly in .env")
        raise

def get_qdrant_client(url: str, api_key: str) -> QdrantClient:
    """
    Initializes and returns the Qdrant client for Qdrant Cloud.
    """
    if not url or not api_key:
        logger.error("QDRANT_ENDPOINT_URL or QDRANT_API_KEY not set in .env file.")
        raise ValueError("Qdrant URL and API Key must be set.")
        
    logger.info(f"Connecting to Qdrant Cloud at {url}")
    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
        )
        # Test connection
        logger.info("Qdrant Cloud connection successful.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant Cloud: {e}")
        raise

def create_qdrant_collection(client: QdrantClient, collection_name: str, embedding_dim: int):
    """
    Creates the Qdrant collection if it doesn't already exist.
    """
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"Collection '{collection_name}' created/recreated successfully.")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise

def prepare_document_for_embedding(doc: Dict[str, Any]) -> str:
    """
    Combines structured JSON fields into a single string for embedding.
    """
    text_parts = [
        f"Patient: {doc.get('patient_information', 'N/A')}",
        f"Complaint: {doc.get('chief_complaint', 'N/A')}",
        f"History: {doc.get('history_of_present_illness', 'N/A')}",
        f"Exposure: {doc.get('exposure_and_epidemiology', 'N/A')}",
        f"Exam: {doc.get('physical_exam', 'N/A')}",
        f"Labs: {doc.get('labs_and_diagnostics', 'N/A')}",
        f"Diagnosis: {doc.get('final_diagnosis', 'N/A')}"
    ]
    return "\n".join(part for part in text_parts if part and 'N/A' not in part)

def main():
    """
    Main function to run the embedding and upsert process.
    """
    logger.info("--- Starting Embedding and Indexing Process ---")
    
    # 1. Initialize models and clients
    try:
        model = get_embedding_model(EMBEDDING_MODEL_NAME, HUGGINGFACETOKEN)
        client = get_qdrant_client(QDRANT_ENDPOINT_URL, QDRANT_API_KEY)
    except Exception:
        logger.error("Failed to initialize models or clients. Exiting.")
        return

    # 2. Create collection
    create_qdrant_collection(client, COLLECTION_NAME, EMBEDDING_DIMENSION)

    # 3. Find JSON files
    json_files = list(INPUT_DIR.glob("*.json"))
    if not json_files:
        logger.warning(f"No .json files found in {INPUT_DIR}. Exiting.")
        return
    
    logger.info(f"Found {len(json_files)} structured JSON files to process.")

    # 4. Process files in batches
    points_batch = []
    batch_size = 32  
    
    for i, file_path in enumerate(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                document_payload = json.load(f)
            
            document_payload['source_filename'] = file_path.name
            text_to_embed = prepare_document_for_embedding(document_payload)
            
            if not text_to_embed:
                logger.warning(f"Skipping {file_path.name}: No text content to embed.")
                continue

            vector = model.encode(text_to_embed).tolist()
            
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=document_payload 
            )
            points_batch.append(point)

            if len(points_batch) >= batch_size or i == len(json_files) - 1:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_batch,
                    wait=True 
                )
                logger.info(f"Upserted batch of {len(points_batch)} points to '{COLLECTION_NAME}'.")
                points_batch = [] # Clear the batch

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")

    logger.info("--- Embedding and Indexing Process Complete ---")

if __name__ == "__main__":
    main()
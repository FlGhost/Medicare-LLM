import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Logger Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# GROBID_CONFIGURATION
GROBID_HOST = os.getenv("GROBID_HOST")

# HUGGINGFACE CONFIGURATION 
HUGGINGFACETOKEN = os.getenv("HUGGINGFACETOKEN")

# QDRANT CLOUD CONFIGURATION
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT_URL = os.getenv("QDRANT_ENDPOINT_URL")

# GEMINI CONFIGURATION
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# WEIGHT & BIASES EVALUATION CONFIGURATION
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
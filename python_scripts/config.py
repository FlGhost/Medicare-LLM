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
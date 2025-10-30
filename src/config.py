# src/config.py
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="us-east-1"
)
S3_BUCKET = os.getenv("S3_BUCKET", "london-realestate-data")
RAW_PREFIX = os.getenv("RAW_PREFIX", "raw")
PROCESSED_PREFIX = os.getenv("PROCESSED_PREFIX", "processed")
GITHUB_RAW_URL = os.getenv("GITHUB_RAW_URL", "https://github.com/Poojitha319/london-rag-agent/blob/main/data/london_properties.csv")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH = os.getenv("INDEX_PATH", "src/data/faiss_index.pkl")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "src/data/embeddings.npy")
PROCESSED_KEY = os.getenv("PROCESSED_KEY", f"{PROCESSED_PREFIX}/clean_properties.csv")
RAW_KEY = os.getenv("RAW_KEY", f"{RAW_PREFIX}/london_properties.csv")

"""
FastAPI Backend for London Real Estate RAG Agentic System
----------------------------------------------------------
Endpoints:
- POST /ingest         → Ingest data from GitHub → S3 /raw
- POST /process        → Clean data → S3 /processed
- GET  /data/sample    → Return sample processed data
- POST /build-index    → Build RAG embeddings
- POST /rag/query      → Retrieve top-k relevant properties
- POST /agent/run      → Agentic reasoning + answer generation
- GET  /health         → Health check
"""
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
import boto3
import io
from src.config import s3, S3_BUCKET
from src.RAG.rag_engine import RAGStore
from src.RAG.agent import SimpleAgent
from src.RAG.utils import load_processed_df



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# -------------------------------------------------
# FastAPI Initialization
# -------------------------------------------------
app = FastAPI(
    title="London Real Estate RAG Chatbot",
    description="AI-driven real estate assistant powered by a Retrieval-Augmented Generation (RAG) pipeline and agentic reasoning.",
    version="1.0.0",
)

rag = RAGStore()

# -------------------------------------------------
# Data Models
# -------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    k: int = 5

class AgentRequest(BaseModel):
    question: str

# -------------------------------------------------
# Health Check
# -------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "London RAG API", "version": "1.0"}

# -------------------------------------------------
# Data Ingestion (GitHub → S3 /raw)
# -------------------------------------------------
@app.post("/ingest")
def ingest_data(file_url: str):
    """
    Download a CSV file from GitHub and upload it to S3 (/raw).
    Example:
    {
      "file_url": "https://raw.githubusercontent.com/username/repo/main/london_properties.csv"
    }
    """

    try:
        response = requests.get(file_url)
        response.raise_for_status()
        csv_bytes = response.content
        s3.put_object(
            Bucket=S3_BUCKET,
            Key="raw/london_properties.csv",
            Body=csv_bytes,
            ContentType="text/csv",
        )
        return {"message": "✅ File successfully ingested to S3 /raw"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting data: {e}")

# -------------------------------------------------
# Data Processing (/raw → /processed)
# -------------------------------------------------
@app.post("/process")
def process_data():
    """
    Clean and standardize raw data, then store in S3 /processed.
    """
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key="raw/london_properties.csv")
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))

        # Basic cleaning rules
        df = df.dropna(subset=["price", "address", "borough"])
        df["borough"] = df["borough"].str.title().str.strip()
        df["property_type"] = df["property_type"].str.title().str.strip()
        df["postcode"] = df["postcode"].str.replace(" ", "").str.upper()

        def normalize_price(val):
            if isinstance(val, str):
                val = val.replace("£", "").replace(",", "").strip().lower()
                if "k" in val:
                    val = float(val.replace("k", "")) * 1000
            return float(val)
        df["price"] = df["price"].apply(normalize_price)
        df = df.sort_values(by=["borough", "price"]).reset_index(drop=True)

        # Upload cleaned CSV to S3 /processed
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key="processed/clean_properties.csv",
            Body=csv_buffer.getvalue(),
            ContentType="text/csv",
        )

        return {"message": "Data cleaned and uploaded to S3 /processed", "rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

# -------------------------------------------------
# View Sample Processed Data
# -------------------------------------------------
@app.get("/data/sample")
def get_sample_data(limit: int = 5):
    """
    Returns sample properties from processed dataset.
    """
    try:
        df = load_processed_df()
        return df.head(limit).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to fetch sample data: {e}")

# -------------------------------------------------
# Build RAG Index
# -------------------------------------------------
@app.post("/build-index")
def build_index():
    """
    Build FAISS embedding index from cleaned data in S3 /processed.
    """
    try:
        df = load_processed_df()
        info = rag.build_index(df)
        return {"message": "RAG index built successfully", "details": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index build failed: {e}")

# -------------------------------------------------
# RAG Query Endpoint
# -------------------------------------------------
@app.post("/rag/query")
def rag_query(req: QueryRequest):
    """
    Retrieve top-k relevant properties using RAG search.
    """
    try:
        if not rag.index:
            rag.load_index()
        results = rag.query(req.question, k=req.k)
        return {"query": req.question, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")

# -------------------------------------------------
# Agentic Reasoning Endpoint
# -------------------------------------------------
@app.post("/agent/run")
def agent_run(req: AgentRequest):
    """
    Run an agentic decision loop: clarify → plan → execute → respond.
    """
    try:
        if rag.df is None:
            df = load_processed_df()
            rag.df = df
            rag.load_index()
        agent = SimpleAgent(rag)
        result = agent.run(req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {e}")

# -------------------------------------------------
# Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

# src/ingest.py
import os
import boto3
import requests
from botocore.exceptions import ClientError
from src.config import S3_BUCKET, RAW_KEY, GITHUB_RAW_URL, AWS_REGION, s3


def fetch_from_github(raw_url: str) -> bytes:
    resp = requests.get(raw_url, timeout=20)
    resp.raise_for_status()
    return resp.content

def upload_to_s3(key: str, content: bytes, content_type="text/csv"):
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=content, ContentType=content_type)
        print(f"Uploaded to s3://{S3_BUCKET}/{key}")
    except ClientError as e:
        print("S3 upload error:", e)
        raise

def run_ingest():
    if not GITHUB_RAW_URL:
        raise ValueError("GITHUB_RAW_URL not configured in env")
    csv_bytes = fetch_from_github(GITHUB_RAW_URL)
    upload_to_s3(RAW_KEY, csv_bytes)

if __name__ == "__main__":
    run_ingest()

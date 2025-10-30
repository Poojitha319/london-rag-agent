import io
import pandas as pd
from src.config import s3, S3_BUCKET

def load_processed_df(key="processed/clean_properties.csv"):
    """Load processed CSV from S3"""
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    return df

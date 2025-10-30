# src/process.py
import io
import re
import boto3
import pandas as pd
from botocore.exceptions import ClientError
from src.config import S3_BUCKET, RAW_KEY, PROCESSED_KEY, AWS_REGION, s3




def download_raw():
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=RAW_KEY)
        content = resp["Body"].read()
        # Check if we got HTML instead of CSV
        if content.startswith(b'<!DOCTYPE html>'):
            raise ValueError("Received HTML instead of CSV. Please check your AWS credentials and S3 bucket permissions.")
        return content
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            raise ValueError(f"The S3 bucket '{S3_BUCKET}' does not exist.")
        elif e.response['Error']['Code'] == 'NoSuchKey':
            raise ValueError(f"The file '{RAW_KEY}' does not exist in bucket '{S3_BUCKET}'.")
        else:
            raise ValueError(f"AWS Error: {str(e)}")

def standardize_postcode(pc: str) -> str:
    if not isinstance(pc, str): return ""
    pc = pc.strip().upper()
    pc = re.sub(r'\s+', ' ', pc)
    return pc

def parse_price(p):
    if pd.isna(p): return None
    s = str(p).strip().lower().replace(',', '').replace('£', '')
    # handle "k" notation
    m = re.match(r'^(\d+(?:\.\d+)?)k$', s)
    if m:
        return int(float(m.group(1)) * 1000)
    try:
        return int(float(s))
    except:
        return None

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # key columns: price, address, borough
    df = df.copy()
    df['price_num'] = df['price'].apply(parse_price)
    df['borough'] = df['borough'].astype(str).str.title().str.strip()
    df['property_type'] = df['property_type'].astype(str).str.title().str.strip()
    df['postcode'] = df['postcode'].apply(standardize_postcode)
    # drop rows lacking critical info
    df = df.dropna(subset=['price_num', 'address', 'borough'])
    # ensure bedrooms numeric
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce').fillna(0).astype(int)
    # reorder and select columns
    keep = ['property_id','address','borough','postcode','property_type','bedrooms','price_num','listing_date','agent_name']
    df = df[keep]
    df = df.rename(columns={'price_num': 'price'})
    return df

def upload_processed(df):
    out = df.to_csv(index=False).encode('utf-8')
    s3.put_object(Bucket=S3_BUCKET, Key=PROCESSED_KEY, Body=out, ContentType='text/csv')
    print("Uploaded processed to s3://{}/{}".format(S3_BUCKET, PROCESSED_KEY))

def run_process():
    # Temporary: Use local file instead of S3
    try:
        df = pd.read_csv("data/london_properties.csv", encoding='utf-8')
        cleaned = clean_df(df)
        upload_processed(cleaned)
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    run_process()

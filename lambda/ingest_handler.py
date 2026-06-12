"""
Ingest Lambda — triggered by S3 ObjectCreated events.
Downloads the PDF, chunks it, embeds with Titan, upserts to Pinecone.

S3 key convention:
  ingest/python/file.pdf       → python_book
  ingest/java/file.pdf         → java_book
  ingest/javascript/file.pdf   → javascript_book
"""
import io
import json
import os

import boto3
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
AWS_REGION       = os.environ.get("AWS_REGION_NAME", "us-east-1")
INDEX_NAME       = "code-docs"
TITAN_MODEL_ID   = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM    = 1024
BATCH_SIZE       = 50

NAMESPACE_MAP = {
    "python":     "python_book",
    "java":       "java_book",
    "javascript": "javascript_book",
}

s3_client       = boto3.client("s3", region_name=AWS_REGION)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
pc              = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index  = pc.Index(INDEX_NAME)


def get_embedding(text: str) -> list:
    response = bedrock_runtime.invoke_model(
        modelId=TITAN_MODEL_ID,
        body=json.dumps({"inputText": text[:8000]}),
    )
    return json.loads(response["body"].read())["embedding"]


def extract_text(pdf_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(
        page.extract_text() or "" for page in reader.pages
    )


def resolve_namespace(s3_key: str) -> str:
    """Derive Pinecone namespace from S3 key path, e.g. ingest/java/book.pdf → java_book."""
    parts = s3_key.lower().split("/")
    for part in parts:
        if part in NAMESPACE_MAP:
            return NAMESPACE_MAP[part]
    raise ValueError(
        f"Cannot determine namespace from key '{s3_key}'. "
        f"Use one of: ingest/python/, ingest/java/, ingest/javascript/"
    )


def lambda_handler(event, context):
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key    = record["s3"]["object"]["key"]
        print(f"Processing s3://{bucket}/{key}")

        try:
            namespace = resolve_namespace(key)
        except ValueError as e:
            print(f"Skipping: {e}")
            continue

        # Download PDF
        obj      = s3_client.get_object(Bucket=bucket, Key=key)
        pdf_bytes = obj["Body"].read()

        # Extract + split
        text   = extract_text(pdf_bytes)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        print(f"  {len(chunks)} chunks → namespace '{namespace}'")

        # Embed + upsert in batches
        filename = key.split("/")[-1].replace(".pdf", "")
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            vectors = []
            for j, chunk in enumerate(batch):
                chunk_id = f"{filename}-{i + j}"
                embedding = get_embedding(chunk)
                vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {"content": chunk, "source": key},
                })
            pinecone_index.upsert(vectors=vectors, namespace=namespace)
            print(f"  Upserted chunks {i + 1}–{i + len(batch)}")

        print(f"  Done: {len(chunks)} chunks in '{namespace}'")

    return {"statusCode": 200, "body": "Ingestion complete"}

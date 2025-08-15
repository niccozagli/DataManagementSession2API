from datetime import timedelta
from pathlib import Path
from typing import Optional
from google.cloud import storage

def make_signed_url(bucket_name: str, rel_path: str, content_type: Optional[str] = None, ttl_minutes: int = 30) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(rel_path)
    if not blob.exists():
        raise FileNotFoundError(rel_path)
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=ttl_minutes),
        method="GET",
        response_disposition=f"attachment; filename={Path(rel_path).name}",
        response_type=content_type,
    )

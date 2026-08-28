from __future__ import annotations

from pathlib import PurePosixPath
from typing import BinaryIO, Optional

from google.cloud import storage


def normalize_prefix(prefix: Optional[str]) -> str:
    if not prefix:
        return ""
    return prefix.strip("/") + "/"


def public_https_url(bucket_name: str, object_name: str) -> str:
    return f"https://storage.googleapis.com/{bucket_name}/{object_name}"


def upload_fileobj_to_gcs(
    fileobj: BinaryIO,
    bucket_name: str,
    prefix: Optional[str],
    filename: str,
    content_type: Optional[str],
) -> dict[str, str | None]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = str(PurePosixPath(normalize_prefix(prefix)) / filename) if prefix else filename
    blob = bucket.blob(blob_name)
    blob.upload_from_file(fileobj, content_type=content_type, rewind=True)
    return {
        "bucket": bucket_name,
        "object_name": blob_name,
        "gcs_uri": f"gs://{bucket_name}/{blob_name}",
        "https_url": public_https_url(bucket_name, blob_name),
        "content_type": content_type,
    }

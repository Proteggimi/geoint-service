"""
Object Storage Service - MinIO/S3 operations
"""
import io
from typing import Optional, BinaryIO
from minio import Minio
from minio.error import S3Error
import structlog

from config import get_settings

logger = structlog.get_logger()
settings = get_settings()

# Global client instance
_minio_client: Optional[Minio] = None


def get_storage_client() -> Minio:
    """Get or create MinIO client"""
    global _minio_client

    if _minio_client is None:
        _minio_client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        logger.info("MinIO client initialized", endpoint=settings.minio_endpoint)

    return _minio_client


async def check_storage_connection() -> bool:
    """Check if storage is accessible"""
    try:
        client = get_storage_client()
        client.list_buckets()
        return True
    except Exception as e:
        logger.error("Storage connection failed", error=str(e))
        return False


async def ensure_buckets_exist():
    """Ensure all required buckets exist"""
    client = get_storage_client()

    buckets = [
        settings.minio_bucket_scenes,
        settings.minio_bucket_cog,
        settings.minio_bucket_thumbnails,
    ]

    for bucket_name in buckets:
        try:
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                logger.info("Created bucket", bucket=bucket_name)
        except S3Error as e:
            logger.error("Failed to create bucket", bucket=bucket_name, error=str(e))
            raise


async def upload_file(
    bucket: str,
    object_name: str,
    data: BinaryIO,
    content_type: str = "application/octet-stream",
    metadata: Optional[dict] = None
) -> str:
    """
    Upload a file to object storage.

    Returns the object URL.
    """
    client = get_storage_client()

    # Get file size
    data.seek(0, 2)  # Seek to end
    size = data.tell()
    data.seek(0)  # Seek back to start

    try:
        client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=data,
            length=size,
            content_type=content_type,
            metadata=metadata
        )

        logger.info("File uploaded", bucket=bucket, object=object_name, size=size)

        # Return internal URL
        return f"s3://{bucket}/{object_name}"

    except S3Error as e:
        logger.error("Upload failed", bucket=bucket, object=object_name, error=str(e))
        raise


async def upload_bytes(
    bucket: str,
    object_name: str,
    data: bytes,
    content_type: str = "application/octet-stream",
    metadata: Optional[dict] = None
) -> str:
    """Upload bytes to storage"""
    return await upload_file(
        bucket=bucket,
        object_name=object_name,
        data=io.BytesIO(data),
        content_type=content_type,
        metadata=metadata
    )


async def get_file(object_url: str) -> bytes:
    """
    Download a file from storage.

    Accepts s3:// URLs or direct bucket/object paths.
    """
    client = get_storage_client()

    # Parse URL
    if object_url.startswith("s3://"):
        parts = object_url[5:].split("/", 1)
        bucket = parts[0]
        object_name = parts[1] if len(parts) > 1 else ""
    else:
        # Assume it's a direct path
        parts = object_url.split("/", 1)
        bucket = parts[0]
        object_name = parts[1] if len(parts) > 1 else ""

    try:
        response = client.get_object(bucket, object_name)
        data = response.read()
        response.close()
        response.release_conn()
        return data

    except S3Error as e:
        logger.error("Download failed", bucket=bucket, object=object_name, error=str(e))
        raise


async def get_presigned_url(
    bucket: str,
    object_name: str,
    expires_hours: int = 1
) -> str:
    """Generate a presigned URL for direct access"""
    from datetime import timedelta

    client = get_storage_client()

    try:
        url = client.presigned_get_object(
            bucket_name=bucket,
            object_name=object_name,
            expires=timedelta(hours=expires_hours)
        )
        return url

    except S3Error as e:
        logger.error("Presigned URL generation failed", error=str(e))
        raise


async def delete_file(bucket: str, object_name: str):
    """Delete a file from storage"""
    client = get_storage_client()

    try:
        client.remove_object(bucket, object_name)
        logger.info("File deleted", bucket=bucket, object=object_name)

    except S3Error as e:
        logger.error("Delete failed", bucket=bucket, object=object_name, error=str(e))
        raise


async def list_objects(bucket: str, prefix: str = "") -> list[dict]:
    """List objects in a bucket with optional prefix"""
    client = get_storage_client()

    objects = []
    try:
        for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
            objects.append({
                "name": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                "etag": obj.etag
            })
        return objects

    except S3Error as e:
        logger.error("List objects failed", bucket=bucket, error=str(e))
        raise

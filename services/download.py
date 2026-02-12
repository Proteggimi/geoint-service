"""
Download Service - Download satellite scenes from providers and convert to COG
"""
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4
import httpx
import structlog

from config import get_settings
from services.storage import upload_file, upload_bytes

logger = structlog.get_logger()
settings = get_settings()


async def enqueue_download(provider: str, scene_id: str) -> str:
    """
    Enqueue a scene download task.
    Returns a task ID for tracking.
    """
    task_id = str(uuid4())

    # In a production environment, this would use Celery
    # For now, we'll run the download directly
    asyncio.create_task(
        download_scene(task_id, provider, scene_id)
    )

    logger.info("Download enqueued",
                task_id=task_id,
                provider=provider,
                scene_id=scene_id)

    return task_id


async def download_scene(task_id: str, provider: str, scene_id: str):
    """
    Download a scene from a provider and convert to COG.
    """
    from database import async_session_maker
    from models.scene import SatelliteScene
    from sqlalchemy import select
    from services.providers import get_provider_factory

    logger.info("Starting scene download",
                task_id=task_id,
                provider=provider,
                scene_id=scene_id)

    try:
        # Get download URL from provider
        factory = get_provider_factory()
        provider_instance = factory.get_provider(provider)

        if not provider_instance:
            raise ValueError(f"Provider {provider} not available")

        download_url = await provider_instance.get_download_url(scene_id)

        if not download_url:
            # For Sentinel via Planetary Computer, the URL is already a COG
            download_url = await _get_planetary_computer_url(scene_id)

        # Download the file
        scene_data = await _download_file(download_url)

        # Save original to storage
        original_path = f"scenes/{provider}/{scene_id}/original.tif"
        original_url = await upload_bytes(
            bucket=settings.minio_bucket_scenes,
            object_name=original_path,
            data=scene_data,
            content_type="image/tiff"
        )

        # Convert to COG if not already
        cog_data = await _convert_to_cog(scene_data)

        cog_path = f"cog/{provider}/{scene_id}/visual.tif"
        cog_url = await upload_bytes(
            bucket=settings.minio_bucket_cog,
            object_name=cog_path,
            data=cog_data,
            content_type="image/tiff"
        )

        # Generate thumbnail
        thumbnail_data = await _generate_thumbnail(scene_data)

        thumbnail_path = f"thumbnails/{provider}/{scene_id}.png"
        thumbnail_url = await upload_bytes(
            bucket=settings.minio_bucket_thumbnails,
            object_name=thumbnail_path,
            data=thumbnail_data,
            content_type="image/png"
        )

        # Update database record
        async with async_session_maker() as session:
            result = await session.execute(
                select(SatelliteScene).where(SatelliteScene.scene_id == scene_id)
            )
            scene = result.scalar_one_or_none()

            if scene:
                scene.storage_path = original_url
                scene.cog_url = cog_url
                scene.thumbnail_url = thumbnail_url
                scene.is_downloaded = True
                scene.is_cog_ready = True
                await session.commit()

        logger.info("Scene download completed",
                    task_id=task_id,
                    scene_id=scene_id)

    except Exception as e:
        logger.error("Scene download failed",
                     task_id=task_id,
                     error=str(e))
        raise


async def _get_planetary_computer_url(scene_id: str) -> str:
    """
    Get COG URL from Microsoft Planetary Computer for Sentinel scenes.
    """
    stac_url = f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/{scene_id}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(stac_url)
        response.raise_for_status()
        data = response.json()

    # Get the visual asset (already a COG)
    assets = data.get("assets", {})

    # Priority: visual > B04 (red) for RGB composite
    if "visual" in assets:
        return assets["visual"]["href"]
    elif "B04" in assets:
        return assets["B04"]["href"]
    else:
        # Return first available asset
        for asset in assets.values():
            if asset.get("type", "").startswith("image/"):
                return asset["href"]

    raise ValueError(f"No suitable asset found for scene {scene_id}")


async def _download_file(url: str, chunk_size: int = 8192) -> bytes:
    """
    Download a file from URL with progress tracking.
    """
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()

            chunks = []
            total_size = 0

            async for chunk in response.aiter_bytes(chunk_size):
                chunks.append(chunk)
                total_size += len(chunk)

                # Log progress for large files
                if total_size % (10 * 1024 * 1024) == 0:  # Every 10MB
                    logger.debug("Download progress", size_mb=total_size / 1024 / 1024)

            return b"".join(chunks)


async def _convert_to_cog(data: bytes) -> bytes:
    """
    Convert raster data to Cloud-Optimized GeoTIFF.
    """
    try:
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles
        from rasterio.io import MemoryFile
        import tempfile

        # Write input to temp file
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_in:
            tmp_in.write(data)
            tmp_in_path = tmp_in.name

        # Create output temp file
        tmp_out_path = tmp_in_path.replace(".tif", "_cog.tif")

        try:
            # Get COG profile
            cog_profile = cog_profiles.get("deflate")

            # Convert to COG
            cog_translate(
                tmp_in_path,
                tmp_out_path,
                cog_profile,
                quiet=True
            )

            # Read result
            with open(tmp_out_path, "rb") as f:
                cog_data = f.read()

            return cog_data

        finally:
            # Cleanup temp files
            for path in [tmp_in_path, tmp_out_path]:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    except ImportError:
        # If rio-cogeo not available, return original data
        logger.warning("rio-cogeo not available, returning original data")
        return data


async def _generate_thumbnail(data: bytes, size: tuple = (256, 256)) -> bytes:
    """
    Generate a thumbnail from raster data.
    """
    try:
        from rasterio.io import MemoryFile
        from PIL import Image
        import numpy as np
        import io

        with MemoryFile(data) as memfile:
            with memfile.open() as src:
                # Read RGB bands (assume 4,3,2 for Sentinel-2)
                if src.count >= 4:
                    rgb = np.dstack([
                        src.read(4),  # Red
                        src.read(3),  # Green
                        src.read(2),  # Blue
                    ])
                elif src.count >= 3:
                    rgb = np.dstack([src.read(1), src.read(2), src.read(3)])
                else:
                    rgb = np.dstack([src.read(1)] * 3)

                # Normalize to 0-255
                rgb = rgb.astype(np.float32)
                for i in range(3):
                    band = rgb[:, :, i]
                    p2, p98 = np.percentile(band[band > 0], (2, 98))
                    band = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                    rgb[:, :, i] = band

                rgb = rgb.astype(np.uint8)

                # Create PIL image and resize
                img = Image.fromarray(rgb)
                img.thumbnail(size, Image.LANCZOS)

                # Save to bytes
                buffer = io.BytesIO()
                img.save(buffer, format="PNG", optimize=True)
                buffer.seek(0)

                return buffer.read()

    except Exception as e:
        logger.error("Thumbnail generation failed", error=str(e))

        # Return placeholder thumbnail
        from PIL import Image
        import io

        img = Image.new("RGB", size, (30, 41, 59))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.read()


async def download_sentinel_quicklook(scene_id: str) -> bytes:
    """
    Download the quicklook/preview image for a Sentinel scene.
    Much faster than downloading the full scene.
    """
    # Planetary Computer provides rendered previews
    preview_url = f"https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?collection=sentinel-2-l2a&item={scene_id}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(preview_url)
            response.raise_for_status()
            return response.content
    except Exception as e:
        logger.error("Failed to download quicklook", scene_id=scene_id, error=str(e))
        raise

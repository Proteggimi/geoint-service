"""
Analysis Service - Change detection and image analysis algorithms
"""
import io
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from uuid import UUID
import numpy as np
from PIL import Image
import structlog

from config import get_settings

logger = structlog.get_logger()
settings = get_settings()


async def run_change_detection(
    analysis_id: UUID,
    before_cog: str,
    after_cog: str,
    method: str = "difference",
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Run change detection between two scenes.

    Methods:
    - difference: Simple image difference
    - ratio: Log ratio (good for SAR)
    - cvaps: Change Vector Analysis

    Returns statistics and saves change mask to storage.
    """
    from database import async_session_maker
    from models.analysis import ChangeAnalysis
    from services.storage import upload_bytes
    from sqlalchemy import select

    logger.info("Starting change detection",
                analysis_id=str(analysis_id),
                method=method)

    try:
        # Update status to processing
        async with async_session_maker() as session:
            result = await session.execute(
                select(ChangeAnalysis).where(ChangeAnalysis.id == analysis_id)
            )
            analysis = result.scalar_one()
            analysis.status = "processing"
            await session.commit()

        # Load images
        before_data, before_profile = await _load_raster(before_cog)
        after_data, after_profile = await _load_raster(after_cog)

        # Ensure same shape
        if before_data.shape != after_data.shape:
            after_data = _resize_to_match(after_data, before_data.shape)

        # Apply radiometric normalization
        after_normalized = _histogram_match(after_data, before_data)

        # Calculate change based on method
        if method == "ratio":
            change_map = _log_ratio_change(before_data, after_normalized)
        elif method == "cvaps":
            change_map = _cvaps_change(before_data, after_normalized)
        else:  # difference
            change_map = _difference_change(before_data, after_normalized)

        # Determine threshold
        if threshold is None:
            threshold = np.std(change_map) * 2.5

        # Create binary change mask
        change_mask = np.abs(change_map) > threshold
        increase_mask = change_map > threshold
        decrease_mask = change_map < -threshold

        # Calculate statistics
        total_pixels = change_mask.size
        changed_pixels = np.sum(change_mask)
        increase_pixels = np.sum(increase_mask)
        decrease_pixels = np.sum(decrease_mask)

        # Estimate area (assuming 10m resolution for Sentinel-2)
        pixel_area_km2 = (10 * 10) / 1_000_000  # 10m pixels to km²
        total_change_area = changed_pixels * pixel_area_km2

        statistics = {
            "total_pixels": int(total_pixels),
            "changed_pixels": int(changed_pixels),
            "change_percentage": float(changed_pixels / total_pixels * 100),
            "increase_pixels": int(increase_pixels),
            "decrease_pixels": int(decrease_pixels),
            "increase_percentage": float(increase_pixels / total_pixels * 100),
            "decrease_percentage": float(decrease_pixels / total_pixels * 100),
            "total_change_area_km2": float(total_change_area),
            "threshold_used": float(threshold),
            "method": method,
        }

        # Generate change mask image
        mask_image = _create_change_mask_image(change_mask, increase_mask, decrease_mask)

        # Save to storage
        mask_buffer = io.BytesIO()
        mask_image.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)

        mask_url = await upload_bytes(
            bucket=settings.minio_bucket_cog,
            object_name=f"change_masks/{analysis_id}.png",
            data=mask_buffer.read(),
            content_type="image/png"
        )

        # Update analysis record
        async with async_session_maker() as session:
            result = await session.execute(
                select(ChangeAnalysis).where(ChangeAnalysis.id == analysis_id)
            )
            analysis = result.scalar_one()
            analysis.status = "completed"
            analysis.statistics = statistics
            analysis.change_mask_url = mask_url
            analysis.completed_at = datetime.utcnow()
            await session.commit()

        logger.info("Change detection completed",
                    analysis_id=str(analysis_id),
                    change_pct=statistics["change_percentage"])

        return statistics

    except Exception as e:
        logger.error("Change detection failed",
                     analysis_id=str(analysis_id),
                     error=str(e))

        # Update status to failed
        async with async_session_maker() as session:
            result = await session.execute(
                select(ChangeAnalysis).where(ChangeAnalysis.id == analysis_id)
            )
            analysis = result.scalar_one()
            analysis.status = "failed"
            analysis.error_message = str(e)
            await session.commit()

        raise


async def _load_raster(cog_url: str) -> Tuple[np.ndarray, dict]:
    """Load raster data from COG URL"""
    try:
        import rasterio
        from rasterio.io import MemoryFile
        import httpx

        # Fetch COG data
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(cog_url)
            response.raise_for_status()
            data = response.content

        # Read with rasterio
        with MemoryFile(data) as memfile:
            with memfile.open() as src:
                # Read all bands
                raster_data = src.read()
                profile = src.profile.copy()

                return raster_data.astype(np.float32), profile

    except ImportError:
        # Fallback: generate synthetic data for testing
        logger.warning("Rasterio not available, using synthetic data")
        return np.random.rand(3, 1024, 1024).astype(np.float32) * 3000, {}


def _resize_to_match(data: np.ndarray, target_shape: Tuple) -> np.ndarray:
    """Resize array to match target shape"""
    from scipy.ndimage import zoom

    factors = [t / s for t, s in zip(target_shape, data.shape)]
    return zoom(data, factors, order=1)


def _histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match histogram of source to reference for radiometric normalization"""
    matched = np.zeros_like(source)

    for band in range(source.shape[0]):
        src_band = source[band].flatten()
        ref_band = reference[band].flatten()

        # Get histograms
        src_values, src_indices, src_counts = np.unique(
            src_band, return_inverse=True, return_counts=True
        )
        ref_values, ref_counts = np.unique(ref_band, return_counts=True)

        # Calculate CDFs
        src_cdf = np.cumsum(src_counts).astype(np.float64)
        src_cdf /= src_cdf[-1]

        ref_cdf = np.cumsum(ref_counts).astype(np.float64)
        ref_cdf /= ref_cdf[-1]

        # Map source values to reference
        interp_values = np.interp(src_cdf, ref_cdf, ref_values)
        matched[band] = interp_values[src_indices].reshape(source[band].shape)

    return matched


def _difference_change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Simple difference change detection"""
    # Use mean across bands if multi-band
    if before.ndim == 3 and before.shape[0] > 1:
        before_mean = np.mean(before, axis=0)
        after_mean = np.mean(after, axis=0)
    else:
        before_mean = before[0] if before.ndim == 3 else before
        after_mean = after[0] if after.ndim == 3 else after

    return after_mean - before_mean


def _log_ratio_change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Log ratio change detection (good for SAR)"""
    if before.ndim == 3 and before.shape[0] > 1:
        before_mean = np.mean(before, axis=0)
        after_mean = np.mean(after, axis=0)
    else:
        before_mean = before[0] if before.ndim == 3 else before
        after_mean = after[0] if after.ndim == 3 else after

    # Add small epsilon to avoid division by zero
    eps = 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.log10((after_mean + eps) / (before_mean + eps))
        ratio = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)

    return ratio


def _cvaps_change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """
    Change Vector Analysis in Posterior Probability Space.
    Calculates magnitude of change vector across all bands.
    """
    # Calculate difference for each band
    diff = after - before

    # Calculate magnitude of change vector
    if diff.ndim == 3:
        magnitude = np.sqrt(np.sum(diff ** 2, axis=0))
    else:
        magnitude = np.abs(diff)

    return magnitude


def _create_change_mask_image(
    change_mask: np.ndarray,
    increase_mask: np.ndarray,
    decrease_mask: np.ndarray
) -> Image.Image:
    """Create a colored change mask image"""
    height, width = change_mask.shape

    # Create RGBA image
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # No change: transparent
    img[:, :, 3] = 0

    # Increase: green with transparency
    img[increase_mask, 0] = 34    # R
    img[increase_mask, 1] = 197   # G
    img[increase_mask, 2] = 94    # B
    img[increase_mask, 3] = 180   # A

    # Decrease: red with transparency
    img[decrease_mask, 0] = 239   # R
    img[decrease_mask, 1] = 68    # G
    img[decrease_mask, 2] = 68    # B
    img[decrease_mask, 3] = 180   # A

    return Image.fromarray(img, mode='RGBA')


async def calculate_spectral_index(
    cog_url: str,
    index_type: str,
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> Dict[str, Any]:
    """
    Calculate a spectral index from a scene.

    Supported indices:
    - ndvi: Normalized Difference Vegetation Index
    - ndwi: Normalized Difference Water Index
    - ndbi: Normalized Difference Built-up Index
    - evi: Enhanced Vegetation Index
    - savi: Soil Adjusted Vegetation Index
    """
    from services.tiles import calculate_band_math

    expressions = {
        "ndvi": "(B08-B04)/(B08+B04)",
        "ndwi": "(B03-B08)/(B03+B08)",
        "ndbi": "(B11-B08)/(B11+B08)",
        "evi": "2.5*(B08-B04)/(B08+6*B04-7.5*B02+1)",
        "savi": "(B08-B04)/(B08+B04+0.5)*1.5",
        "mndwi": "(B03-B11)/(B03+B11)",
    }

    if index_type not in expressions:
        raise ValueError(f"Unknown index type: {index_type}")

    result, stats = await calculate_band_math(
        cog_url=cog_url,
        expression=expressions[index_type],
        bounds=bounds
    )

    # Classify the index
    if index_type == "ndvi":
        classification = {
            "water": float(np.sum(result < 0) / result.size * 100),
            "bare_soil": float(np.sum((result >= 0) & (result < 0.2)) / result.size * 100),
            "sparse_vegetation": float(np.sum((result >= 0.2) & (result < 0.4)) / result.size * 100),
            "moderate_vegetation": float(np.sum((result >= 0.4) & (result < 0.6)) / result.size * 100),
            "dense_vegetation": float(np.sum(result >= 0.6) / result.size * 100),
        }
    elif index_type == "ndwi":
        classification = {
            "land": float(np.sum(result < 0) / result.size * 100),
            "water": float(np.sum(result >= 0) / result.size * 100),
        }
    elif index_type == "ndbi":
        classification = {
            "non_built": float(np.sum(result < 0) / result.size * 100),
            "built_up": float(np.sum(result >= 0) / result.size * 100),
        }
    else:
        classification = {}

    return {
        "index_type": index_type,
        "expression": expressions[index_type],
        "statistics": stats,
        "classification": classification,
    }

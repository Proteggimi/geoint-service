"""
Tile Service - Generate map tiles from Cloud-Optimized GeoTIFFs
Uses rasterio and rio-tiler for efficient tile generation
"""
import io
from typing import Optional, Tuple, List
from functools import lru_cache
import numpy as np
from PIL import Image
import httpx
import structlog

from config import get_settings

logger = structlog.get_logger()
settings = get_settings()


# Band presets for different visualizations
BAND_PRESETS = {
    "true_color": {"bands": [4, 3, 2], "name": "True Color (RGB)"},
    "false_color": {"bands": [8, 4, 3], "name": "False Color (NIR)"},
    "agriculture": {"bands": [11, 8, 2], "name": "Agriculture"},
    "ndvi": {"expression": "(B08-B04)/(B08+B04)", "name": "NDVI"},
    "ndwi": {"expression": "(B03-B08)/(B03+B08)", "name": "NDWI"},
    "ndbi": {"expression": "(B11-B08)/(B11+B08)", "name": "NDBI"},
}

# Colormaps for index visualization
COLORMAPS = {
    "rdylgn": [(165, 0, 38), (215, 48, 39), (244, 109, 67), (253, 174, 97),
               (254, 224, 139), (255, 255, 191), (217, 239, 139), (166, 217, 106),
               (102, 189, 99), (26, 152, 80), (0, 104, 55)],
    "blues": [(247, 251, 255), (222, 235, 247), (198, 219, 239), (158, 202, 225),
              (107, 174, 214), (66, 146, 198), (33, 113, 181), (8, 81, 156), (8, 48, 107)],
    "reds": [(255, 245, 240), (254, 224, 210), (252, 187, 161), (252, 146, 114),
             (251, 106, 74), (239, 59, 44), (203, 24, 29), (165, 15, 21), (103, 0, 13)],
    "viridis": [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
                (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
                (180, 222, 44), (253, 231, 37)],
}


def parse_bands(bands_str: str) -> List[int]:
    """Parse band string like '4,3,2' to list of integers"""
    try:
        return [int(b.strip()) for b in bands_str.split(',')]
    except ValueError:
        return [4, 3, 2]  # Default to RGB


def parse_rescale(rescale_str: str) -> Tuple[float, float]:
    """Parse rescale string like '0,3000' to tuple"""
    try:
        parts = rescale_str.split(',')
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return 0.0, 3000.0


def apply_colormap(data: np.ndarray, colormap_name: str = "rdylgn") -> np.ndarray:
    """Apply a colormap to single-band data"""
    colormap = COLORMAPS.get(colormap_name, COLORMAPS["rdylgn"])

    # Normalize data to 0-1
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    if data_max - data_min > 0:
        normalized = (data - data_min) / (data_max - data_min)
    else:
        normalized = np.zeros_like(data)

    # Map to colormap indices
    indices = (normalized * (len(colormap) - 1)).astype(int)
    indices = np.clip(indices, 0, len(colormap) - 1)

    # Create RGB output
    rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
    for i, color in enumerate(colormap):
        mask = indices == i
        rgb[mask] = color

    return rgb


def rescale_intensity(data: np.ndarray, in_range: Tuple[float, float],
                       out_range: Tuple[int, int] = (0, 255)) -> np.ndarray:
    """Rescale data intensity to output range"""
    in_min, in_max = in_range
    out_min, out_max = out_range

    # Clip to input range
    data = np.clip(data, in_min, in_max)

    # Scale to output range
    if in_max - in_min > 0:
        scaled = (data - in_min) / (in_max - in_min)
        scaled = scaled * (out_max - out_min) + out_min
    else:
        scaled = np.full_like(data, out_min)

    return scaled.astype(np.uint8)


async def get_tile_from_cog(
    cog_url: str,
    z: int,
    x: int,
    y: int,
    bands: str = "4,3,2",
    rescale: str = "0,3000",
    colormap: Optional[str] = None,
    tile_size: int = 256
) -> bytes:
    """
    Generate a map tile from a Cloud-Optimized GeoTIFF.

    Uses HTTP range requests for efficient access to COG overviews.
    """
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        from rio_tiler.io import COGReader
        from rio_tiler.models import ImageData
    except ImportError:
        # Fallback to simple tile generation without rio-tiler
        return await _generate_fallback_tile(cog_url, z, x, y, tile_size)

    band_list = parse_bands(bands)
    rescale_range = parse_rescale(rescale)

    try:
        with COGReader(cog_url) as cog:
            # Read tile
            img = cog.tile(x, y, z, indexes=band_list, tilesize=tile_size)

            # Get data as numpy array
            data = img.data

            if colormap and len(band_list) == 1:
                # Single band with colormap
                rgb = apply_colormap(data[0], colormap)
            else:
                # Multi-band RGB
                rgb = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                for i in range(min(3, len(band_list))):
                    rgb[:, :, i] = rescale_intensity(data[i], rescale_range)

            # Handle alpha/mask
            if img.mask is not None:
                alpha = (img.mask * 255).astype(np.uint8)
                rgba = np.dstack([rgb, alpha])
                image = Image.fromarray(rgba, mode='RGBA')
            else:
                image = Image.fromarray(rgb, mode='RGB')

            # Save to bytes
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', optimize=True)
            buffer.seek(0)

            return buffer.read()

    except Exception as e:
        logger.error("Tile generation failed", error=str(e), cog_url=cog_url)
        return await _generate_error_tile(tile_size, str(e))


async def _generate_fallback_tile(cog_url: str, z: int, x: int, y: int,
                                   tile_size: int = 256) -> bytes:
    """Generate a placeholder tile when rio-tiler is not available"""
    # Create a simple gradient tile as placeholder
    img = Image.new('RGBA', (tile_size, tile_size), (30, 41, 59, 200))

    # Add some visual indication
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((10, tile_size//2 - 10), f"z={z}", fill=(100, 116, 139))
    draw.text((10, tile_size//2 + 5), f"x={x}, y={y}", fill=(100, 116, 139))

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.read()


async def _generate_error_tile(tile_size: int, error_msg: str = "") -> bytes:
    """Generate an error indicator tile"""
    img = Image.new('RGBA', (tile_size, tile_size), (239, 68, 68, 100))

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.read()


async def generate_preview(
    cog_url: str,
    width: int = 512,
    height: int = 512,
    bands: str = "4,3,2",
    rescale: str = "0,3000"
) -> bytes:
    """
    Generate a preview image of the entire scene.
    """
    try:
        import rasterio
        from rio_tiler.io import COGReader
    except ImportError:
        return await _generate_placeholder_preview(width, height)

    band_list = parse_bands(bands)
    rescale_range = parse_rescale(rescale)

    try:
        with COGReader(cog_url) as cog:
            # Get preview at specified size
            img = cog.preview(
                indexes=band_list,
                max_size=max(width, height)
            )

            data = img.data

            # Create RGB image
            rgb = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
            for i in range(min(3, len(band_list))):
                rgb[:, :, i] = rescale_intensity(data[i], rescale_range)

            image = Image.fromarray(rgb, mode='RGB')
            image = image.resize((width, height), Image.LANCZOS)

            buffer = io.BytesIO()
            image.save(buffer, format='PNG', quality=85)
            buffer.seek(0)

            return buffer.read()

    except Exception as e:
        logger.error("Preview generation failed", error=str(e))
        return await _generate_placeholder_preview(width, height)


async def _generate_placeholder_preview(width: int, height: int) -> bytes:
    """Generate a placeholder preview image"""
    img = Image.new('RGB', (width, height), (30, 41, 59))

    # Add grid pattern
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    for x in range(0, width, 50):
        draw.line([(x, 0), (x, height)], fill=(51, 65, 85), width=1)
    for y in range(0, height, 50):
        draw.line([(0, y), (width, y)], fill=(51, 65, 85), width=1)

    # Add text
    draw.text((width//2 - 60, height//2 - 10), "Preview Unavailable", fill=(100, 116, 139))

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.read()


async def calculate_band_math(
    cog_url: str,
    expression: str,
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[np.ndarray, dict]:
    """
    Calculate band math expression on a COG.

    Expression format: "(B08-B04)/(B08+B04)" for NDVI
    Returns the computed array and statistics.
    """
    try:
        from rio_tiler.io import COGReader
        import re
    except ImportError:
        raise RuntimeError("rio-tiler required for band math")

    # Parse band references from expression
    band_refs = re.findall(r'B(\d{2}|8A)', expression.upper())
    band_map = {}

    # Map band names to indices (Sentinel-2 specific)
    sentinel2_bands = {
        'B01': 1, 'B02': 2, 'B03': 3, 'B04': 4, 'B05': 5,
        'B06': 6, 'B07': 7, 'B08': 8, 'B8A': 9, 'B09': 10,
        'B10': 11, 'B11': 12, 'B12': 13
    }

    with COGReader(cog_url) as cog:
        # Read required bands
        bands_to_read = []
        for ref in band_refs:
            band_name = f"B{ref}"
            if band_name in sentinel2_bands:
                bands_to_read.append(sentinel2_bands[band_name])
                band_map[band_name] = len(bands_to_read) - 1

        if bounds:
            img = cog.part(bounds, indexes=bands_to_read)
        else:
            img = cog.preview(indexes=bands_to_read, max_size=1024)

        data = img.data.astype(np.float32)

        # Build expression with array references
        calc_expr = expression.upper()
        for band_name, idx in band_map.items():
            calc_expr = calc_expr.replace(band_name, f"data[{idx}]")

        # Evaluate expression
        with np.errstate(divide='ignore', invalid='ignore'):
            result = eval(calc_expr)
            result = np.nan_to_num(result, nan=0, posinf=1, neginf=-1)

        # Calculate statistics
        valid_data = result[~np.isnan(result)]
        stats = {
            "min": float(np.min(valid_data)) if len(valid_data) > 0 else 0,
            "max": float(np.max(valid_data)) if len(valid_data) > 0 else 0,
            "mean": float(np.mean(valid_data)) if len(valid_data) > 0 else 0,
            "std": float(np.std(valid_data)) if len(valid_data) > 0 else 0,
        }

        return result, stats

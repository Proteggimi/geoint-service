"""
Tiles Router - Serve COG tiles for visualization
"""
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_session
from models.scene import SatelliteScene
from config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/{scene_id}/{z}/{x}/{y}.png")
async def get_tile(
    scene_id: UUID,
    z: int,
    x: int,
    y: int,
    bands: Optional[str] = Query(default="4,3,2", description="Band indices (e.g., '4,3,2' for RGB)"),
    rescale: Optional[str] = Query(default="0,3000", description="Min,max rescale values"),
    colormap: Optional[str] = Query(default=None, description="Colormap name for single-band"),
    session: AsyncSession = Depends(get_session)
):
    """
    Get a map tile for the specified scene.
    Uses TiTiler to serve Cloud-Optimized GeoTIFF tiles.
    """
    # Get scene from database
    result = await session.execute(
        select(SatelliteScene.cog_url, SatelliteScene.is_cog_ready)
        .where(SatelliteScene.id == scene_id)
    )
    row = result.one_or_none()

    if not row:
        raise HTTPException(status_code=404, detail="Scene not found")

    cog_url, is_cog_ready = row

    if not is_cog_ready or not cog_url:
        raise HTTPException(
            status_code=503,
            detail="Scene not yet processed. Please try again later."
        )

    # Forward to TiTiler
    from services.tiles import get_tile_from_cog

    try:
        tile_bytes = await get_tile_from_cog(
            cog_url=cog_url,
            z=z, x=x, y=y,
            bands=bands,
            rescale=rescale,
            colormap=colormap
        )

        return Response(
            content=tile_bytes,
            media_type="image/png",
            headers={
                "Cache-Control": f"public, max-age={settings.tile_cache_ttl}",
                "X-Scene-ID": str(scene_id)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tile generation failed: {str(e)}")


@router.get("/{scene_id}/tilejson.json")
async def get_tilejson(
    scene_id: UUID,
    bands: Optional[str] = Query(default="4,3,2"),
    session: AsyncSession = Depends(get_session)
):
    """
    Get TileJSON metadata for a scene.
    Used by MapLibre/Leaflet for layer configuration.
    """
    result = await session.execute(
        select(SatelliteScene)
        .where(SatelliteScene.id == scene_id)
    )
    scene = result.scalar_one_or_none()

    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")

    # Get bounds from footprint
    from geoalchemy2.shape import to_shape
    footprint = to_shape(scene.footprint)
    bounds = list(footprint.bounds)  # [minx, miny, maxx, maxy]

    center = footprint.centroid
    center_coords = [center.x, center.y]

    return {
        "tilejson": "3.0.0",
        "name": f"{scene.provider} - {scene.scene_id}",
        "description": f"Acquired: {scene.acquisition_date.isoformat()}",
        "version": "1.0.0",
        "attribution": f"Source: {scene.provider}",
        "scheme": "xyz",
        "tiles": [
            f"/geoint/tiles/{scene_id}/{{z}}/{{x}}/{{y}}.png?bands={bands}"
        ],
        "minzoom": 1,
        "maxzoom": 18,
        "bounds": bounds,
        "center": center_coords + [10],  # [lng, lat, zoom]
    }


@router.get("/{scene_id}/preview.png")
async def get_preview(
    scene_id: UUID,
    width: int = Query(default=512, ge=64, le=2048),
    height: int = Query(default=512, ge=64, le=2048),
    bands: Optional[str] = Query(default="4,3,2"),
    session: AsyncSession = Depends(get_session)
):
    """
    Get a preview image of the entire scene.
    """
    result = await session.execute(
        select(SatelliteScene.cog_url, SatelliteScene.thumbnail_url, SatelliteScene.is_cog_ready)
        .where(SatelliteScene.id == scene_id)
    )
    row = result.one_or_none()

    if not row:
        raise HTTPException(status_code=404, detail="Scene not found")

    cog_url, thumbnail_url, is_cog_ready = row

    # If we have a pre-generated thumbnail, use it
    if thumbnail_url:
        from services.storage import get_file
        thumbnail_bytes = await get_file(thumbnail_url)
        return Response(content=thumbnail_bytes, media_type="image/png")

    # Otherwise generate from COG
    if not is_cog_ready or not cog_url:
        raise HTTPException(status_code=503, detail="Scene not ready")

    from services.tiles import generate_preview

    preview_bytes = await generate_preview(
        cog_url=cog_url,
        width=width,
        height=height,
        bands=bands
    )

    return Response(
        content=preview_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"}
    )


@router.get("/composites")
async def list_composites():
    """
    List available band composite presets.
    """
    return {
        "optical": {
            "true_color": {
                "name": "True Color (RGB)",
                "bands": "4,3,2",
                "description": "Natural colors as seen by human eye"
            },
            "false_color": {
                "name": "False Color (NIR)",
                "bands": "8,4,3",
                "description": "Vegetation appears red, useful for land cover"
            },
            "agriculture": {
                "name": "Agriculture",
                "bands": "11,8,2",
                "description": "Highlights crop health and soil moisture"
            },
            "atmospheric_penetration": {
                "name": "Atmospheric Penetration",
                "bands": "12,11,8a",
                "description": "Reduces atmospheric haze"
            },
            "moisture": {
                "name": "Moisture Index",
                "bands": "8a,11,12",
                "description": "Shows moisture content in vegetation and soil"
            },
            "urban": {
                "name": "Urban",
                "bands": "12,11,4",
                "description": "Highlights urban areas and bare soil"
            }
        },
        "indices": {
            "ndvi": {
                "name": "NDVI",
                "expression": "(B08-B04)/(B08+B04)",
                "description": "Normalized Difference Vegetation Index",
                "colormap": "rdylgn"
            },
            "ndwi": {
                "name": "NDWI",
                "expression": "(B03-B08)/(B03+B08)",
                "description": "Normalized Difference Water Index",
                "colormap": "blues"
            },
            "ndbi": {
                "name": "NDBI",
                "expression": "(B11-B08)/(B11+B08)",
                "description": "Normalized Difference Built-up Index",
                "colormap": "reds"
            }
        },
        "sar": {
            "vv": {
                "name": "VV Polarization",
                "bands": "VV",
                "description": "Vertical-Vertical polarization"
            },
            "vh": {
                "name": "VH Polarization",
                "bands": "VH",
                "description": "Vertical-Horizontal polarization"
            },
            "vv_vh_ratio": {
                "name": "VV/VH Ratio",
                "expression": "VV/VH",
                "description": "Polarization ratio for surface classification"
            }
        }
    }

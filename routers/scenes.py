"""
Scenes Router - Search and manage satellite scenes
"""
from datetime import datetime
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from geoalchemy2.functions import ST_Intersects, ST_GeomFromGeoJSON

from database import get_session
from models.scene import SatelliteScene
from services.providers import get_provider_factory

router = APIRouter()


# Pydantic schemas
class GeoJSONPolygon(BaseModel):
    type: str = "Polygon"
    coordinates: List[List[List[float]]]


class SceneSearchRequest(BaseModel):
    """Request body for scene search"""
    aoi: GeoJSONPolygon = Field(..., description="Area of interest as GeoJSON Polygon")
    start_date: datetime = Field(..., description="Start of date range")
    end_date: datetime = Field(..., description="End of date range")
    providers: Optional[List[str]] = Field(
        default=["sentinel"],
        description="Satellite providers to search"
    )
    max_cloud_cover: Optional[float] = Field(
        default=30.0,
        ge=0,
        le=100,
        description="Maximum cloud cover percentage"
    )
    limit: Optional[int] = Field(default=50, ge=1, le=200)


class SceneResponse(BaseModel):
    """Scene metadata response"""
    id: UUID
    provider: str
    scene_id: str
    acquisition_date: datetime
    cloud_cover: Optional[float]
    resolution_m: Optional[float]
    sensor: Optional[str]
    thumbnail_url: Optional[str]
    cog_url: Optional[str]
    is_downloaded: bool
    is_cog_ready: bool

    class Config:
        from_attributes = True


class SceneSearchResponse(BaseModel):
    """Search results response"""
    total: int
    scenes: List[SceneResponse]


@router.post("/search", response_model=SceneSearchResponse)
async def search_scenes(
    request: SceneSearchRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Search for satellite scenes in the given AOI and date range.
    Queries configured providers and returns matching scenes.
    """
    import json

    # First, search in local database
    aoi_geojson = json.dumps(request.aoi.model_dump())

    query = select(SatelliteScene).where(
        ST_Intersects(
            SatelliteScene.footprint,
            ST_GeomFromGeoJSON(aoi_geojson)
        ),
        SatelliteScene.acquisition_date >= request.start_date,
        SatelliteScene.acquisition_date <= request.end_date,
    )

    if request.max_cloud_cover is not None:
        query = query.where(SatelliteScene.cloud_cover <= request.max_cloud_cover)

    if request.providers:
        query = query.where(SatelliteScene.provider.in_(request.providers))

    query = query.order_by(SatelliteScene.acquisition_date.desc()).limit(request.limit)

    result = await session.execute(query)
    local_scenes = result.scalars().all()

    # If we have enough local results, return them
    if len(local_scenes) >= request.limit:
        return SceneSearchResponse(
            total=len(local_scenes),
            scenes=[SceneResponse.model_validate(s) for s in local_scenes]
        )

    # Otherwise, also query external providers
    provider_factory = get_provider_factory()
    external_scenes = []

    for provider_name in request.providers or ["sentinel"]:
        provider = provider_factory.get_provider(provider_name)
        if provider and provider.is_enabled():
            try:
                scenes = await provider.search(
                    aoi=request.aoi.model_dump(),
                    start_date=request.start_date,
                    end_date=request.end_date,
                    max_cloud_cover=request.max_cloud_cover,
                    limit=request.limit - len(local_scenes)
                )
                external_scenes.extend(scenes)
            except Exception as e:
                # Log error but continue with other providers
                pass

    # Combine and deduplicate
    all_scenes = list(local_scenes) + external_scenes
    unique_scenes = {s.scene_id if hasattr(s, 'scene_id') else s['scene_id']: s for s in all_scenes}

    return SceneSearchResponse(
        total=len(unique_scenes),
        scenes=list(unique_scenes.values())[:request.limit]
    )


@router.get("/{scene_id}", response_model=SceneResponse)
async def get_scene(
    scene_id: UUID,
    session: AsyncSession = Depends(get_session)
):
    """Get a specific scene by ID"""
    result = await session.execute(
        select(SatelliteScene).where(SatelliteScene.id == scene_id)
    )
    scene = result.scalar_one_or_none()

    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")

    return SceneResponse.model_validate(scene)


@router.post("/{scene_id}/download")
async def request_download(
    scene_id: str,
    provider: str = Query(..., description="Provider name"),
    session: AsyncSession = Depends(get_session)
):
    """
    Request download of a scene from the provider.
    Returns a task ID for tracking the download progress.
    """
    from services.download import enqueue_download

    task_id = await enqueue_download(provider, scene_id)

    return {
        "status": "queued",
        "task_id": task_id,
        "message": f"Download queued for {provider}:{scene_id}"
    }


@router.get("/{scene_id}/bands")
async def get_scene_bands(
    scene_id: UUID,
    session: AsyncSession = Depends(get_session)
):
    """Get available bands for a scene"""
    result = await session.execute(
        select(SatelliteScene.bands, SatelliteScene.sensor, SatelliteScene.provider)
        .where(SatelliteScene.id == scene_id)
    )
    row = result.one_or_none()

    if not row:
        raise HTTPException(status_code=404, detail="Scene not found")

    bands, sensor, provider = row

    # Default band info if not stored
    if not bands:
        if provider == "sentinel" and sensor in ["S2A", "S2B", "MSI"]:
            bands = [
                {"name": "B01", "wavelength": "443nm", "resolution": 60, "description": "Coastal aerosol"},
                {"name": "B02", "wavelength": "490nm", "resolution": 10, "description": "Blue"},
                {"name": "B03", "wavelength": "560nm", "resolution": 10, "description": "Green"},
                {"name": "B04", "wavelength": "665nm", "resolution": 10, "description": "Red"},
                {"name": "B05", "wavelength": "705nm", "resolution": 20, "description": "Vegetation Red Edge"},
                {"name": "B06", "wavelength": "740nm", "resolution": 20, "description": "Vegetation Red Edge"},
                {"name": "B07", "wavelength": "783nm", "resolution": 20, "description": "Vegetation Red Edge"},
                {"name": "B08", "wavelength": "842nm", "resolution": 10, "description": "NIR"},
                {"name": "B8A", "wavelength": "865nm", "resolution": 20, "description": "Vegetation Red Edge"},
                {"name": "B09", "wavelength": "945nm", "resolution": 60, "description": "Water vapour"},
                {"name": "B10", "wavelength": "1375nm", "resolution": 60, "description": "SWIR - Cirrus"},
                {"name": "B11", "wavelength": "1610nm", "resolution": 20, "description": "SWIR"},
                {"name": "B12", "wavelength": "2190nm", "resolution": 20, "description": "SWIR"},
            ]

    return {
        "scene_id": str(scene_id),
        "provider": provider,
        "sensor": sensor,
        "bands": bands or [],
        "presets": {
            "true_color": ["B04", "B03", "B02"],
            "false_color": ["B08", "B04", "B03"],
            "agriculture": ["B11", "B08", "B02"],
            "ndvi": ["B08", "B04"],
        }
    }

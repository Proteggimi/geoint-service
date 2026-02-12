"""
Analysis Router - Change detection and band math
"""
from datetime import datetime
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_session
from models.analysis import ChangeAnalysis
from models.scene import SatelliteScene

router = APIRouter()


class GeoJSONPolygon(BaseModel):
    type: str = "Polygon"
    coordinates: List[List[List[float]]]


class ChangeDetectionRequest(BaseModel):
    """Request for change detection analysis"""
    before_scene_id: UUID = Field(..., description="ID of the 'before' scene")
    after_scene_id: UUID = Field(..., description="ID of the 'after' scene")
    aoi: Optional[GeoJSONPolygon] = Field(
        default=None,
        description="Area of interest (defaults to scene intersection)"
    )
    method: str = Field(
        default="difference",
        description="Detection method: 'difference', 'ratio', 'cvaps'"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Custom threshold for change classification"
    )


class ChangeDetectionResponse(BaseModel):
    """Response with analysis results"""
    id: UUID
    status: str
    before_scene_id: UUID
    after_scene_id: UUID
    method: str
    statistics: Optional[dict] = None
    change_mask_url: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class BandMathRequest(BaseModel):
    """Request for band math calculation"""
    scene_id: UUID
    expression: str = Field(
        ...,
        description="Band math expression, e.g., '(B08-B04)/(B08+B04)' for NDVI"
    )
    aoi: Optional[GeoJSONPolygon] = None
    colormap: Optional[str] = Field(default="rdylgn")
    rescale: Optional[str] = Field(default="-1,1")


@router.post("/change-detection", response_model=ChangeDetectionResponse)
async def create_change_detection(
    request: ChangeDetectionRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session)
):
    """
    Start a change detection analysis between two scenes.
    Returns immediately with a task ID for tracking progress.
    """
    import json
    from geoalchemy2.functions import ST_Intersection, ST_AsGeoJSON

    # Verify both scenes exist and are ready
    before_result = await session.execute(
        select(SatelliteScene).where(SatelliteScene.id == request.before_scene_id)
    )
    before_scene = before_result.scalar_one_or_none()

    after_result = await session.execute(
        select(SatelliteScene).where(SatelliteScene.id == request.after_scene_id)
    )
    after_scene = after_result.scalar_one_or_none()

    if not before_scene or not after_scene:
        raise HTTPException(status_code=404, detail="One or both scenes not found")

    if not before_scene.is_cog_ready or not after_scene.is_cog_ready:
        raise HTTPException(
            status_code=400,
            detail="Both scenes must be processed before analysis"
        )

    # Calculate AOI from scene intersection if not provided
    if request.aoi:
        aoi_geojson = json.dumps(request.aoi.model_dump())
    else:
        # Get intersection of both footprints
        intersection_result = await session.execute(
            select(ST_AsGeoJSON(ST_Intersection(
                before_scene.footprint,
                after_scene.footprint
            )))
        )
        aoi_geojson = intersection_result.scalar_one()

    # Create analysis record
    from geoalchemy2.functions import ST_GeomFromGeoJSON

    analysis = ChangeAnalysis(
        before_scene_id=request.before_scene_id,
        after_scene_id=request.after_scene_id,
        aoi=ST_GeomFromGeoJSON(aoi_geojson),
        method=request.method,
        parameters={"threshold": request.threshold} if request.threshold else None,
        status="pending"
    )

    session.add(analysis)
    await session.commit()
    await session.refresh(analysis)

    # Queue background processing
    from services.analysis import run_change_detection
    background_tasks.add_task(
        run_change_detection,
        analysis_id=analysis.id,
        before_cog=before_scene.cog_url,
        after_cog=after_scene.cog_url,
        method=request.method,
        threshold=request.threshold
    )

    return ChangeDetectionResponse.model_validate(analysis)


@router.get("/change-detection/{analysis_id}", response_model=ChangeDetectionResponse)
async def get_change_detection(
    analysis_id: UUID,
    session: AsyncSession = Depends(get_session)
):
    """Get the status and results of a change detection analysis"""
    result = await session.execute(
        select(ChangeAnalysis).where(ChangeAnalysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return ChangeDetectionResponse.model_validate(analysis)


@router.get("/change-detection")
async def list_change_detections(
    status: Optional[str] = None,
    limit: int = 20,
    session: AsyncSession = Depends(get_session)
):
    """List all change detection analyses"""
    query = select(ChangeAnalysis).order_by(ChangeAnalysis.created_at.desc())

    if status:
        query = query.where(ChangeAnalysis.status == status)

    query = query.limit(limit)

    result = await session.execute(query)
    analyses = result.scalars().all()

    return {
        "total": len(analyses),
        "analyses": [ChangeDetectionResponse.model_validate(a) for a in analyses]
    }


@router.post("/band-math")
async def calculate_band_math(
    request: BandMathRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Calculate a band math expression on a scene.
    Returns tile URL for visualization.
    """
    # Validate scene exists
    result = await session.execute(
        select(SatelliteScene.cog_url, SatelliteScene.is_cog_ready)
        .where(SatelliteScene.id == request.scene_id)
    )
    row = result.one_or_none()

    if not row:
        raise HTTPException(status_code=404, detail="Scene not found")

    cog_url, is_cog_ready = row

    if not is_cog_ready:
        raise HTTPException(status_code=400, detail="Scene not ready")

    # Parse expression to validate
    valid_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                   "B08", "B8A", "B09", "B10", "B11", "B12", "VV", "VH"]

    import re
    bands_used = re.findall(r'B\d{2}|B8A|VV|VH', request.expression.upper())

    if not bands_used:
        raise HTTPException(status_code=400, detail="No valid bands in expression")

    return {
        "scene_id": str(request.scene_id),
        "expression": request.expression,
        "bands_used": bands_used,
        "tile_url": f"/geoint/tiles/{request.scene_id}/{{z}}/{{x}}/{{y}}.png?expression={request.expression}&colormap={request.colormap}&rescale={request.rescale}",
        "colormap": request.colormap,
        "rescale": request.rescale
    }


@router.get("/indices")
async def list_available_indices():
    """List all available spectral indices with formulas"""
    return {
        "vegetation": [
            {
                "name": "NDVI",
                "full_name": "Normalized Difference Vegetation Index",
                "expression": "(B08-B04)/(B08+B04)",
                "range": [-1, 1],
                "interpretation": "Values > 0.4 indicate healthy vegetation"
            },
            {
                "name": "EVI",
                "full_name": "Enhanced Vegetation Index",
                "expression": "2.5*(B08-B04)/(B08+6*B04-7.5*B02+1)",
                "range": [-1, 1],
                "interpretation": "More sensitive in high biomass areas"
            },
            {
                "name": "SAVI",
                "full_name": "Soil Adjusted Vegetation Index",
                "expression": "(B08-B04)/(B08+B04+0.5)*1.5",
                "range": [-1, 1],
                "interpretation": "Minimizes soil brightness influence"
            }
        ],
        "water": [
            {
                "name": "NDWI",
                "full_name": "Normalized Difference Water Index",
                "expression": "(B03-B08)/(B03+B08)",
                "range": [-1, 1],
                "interpretation": "Positive values indicate water"
            },
            {
                "name": "MNDWI",
                "full_name": "Modified NDWI",
                "expression": "(B03-B11)/(B03+B11)",
                "range": [-1, 1],
                "interpretation": "Better at suppressing built-up noise"
            }
        ],
        "urban": [
            {
                "name": "NDBI",
                "full_name": "Normalized Difference Built-up Index",
                "expression": "(B11-B08)/(B11+B08)",
                "range": [-1, 1],
                "interpretation": "Positive values indicate built-up areas"
            },
            {
                "name": "UI",
                "full_name": "Urban Index",
                "expression": "(B11-B08)/(B11+B08)",
                "range": [-1, 1],
                "interpretation": "Highlights urban areas"
            }
        ],
        "geology": [
            {
                "name": "Clay Ratio",
                "expression": "B11/B12",
                "range": [0, 2],
                "interpretation": "Higher values indicate clay minerals"
            },
            {
                "name": "Ferrous Minerals",
                "expression": "B11/B08",
                "range": [0, 2],
                "interpretation": "Highlights iron-bearing minerals"
            }
        ]
    }

"""
Detections Router - Object detection in satellite imagery
"""
from datetime import datetime
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from geoalchemy2.functions import ST_AsGeoJSON, ST_Within, ST_GeomFromGeoJSON

from database import get_session
from models.detection import ObjectDetection
from models.scene import SatelliteScene

router = APIRouter()


class GeoJSONPolygon(BaseModel):
    type: str = "Polygon"
    coordinates: List[List[List[float]]]


class DetectionRequest(BaseModel):
    """Request to run object detection on a scene"""
    scene_id: UUID
    aoi: Optional[GeoJSONPolygon] = Field(
        default=None,
        description="Limit detection to this area"
    )
    detection_types: List[str] = Field(
        default=["vehicle", "ship", "aircraft", "building"],
        description="Types of objects to detect"
    )
    confidence_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for detections"
    )


class DetectionResult(BaseModel):
    """Single detection result"""
    id: int
    detection_type: str
    class_name: str
    confidence: float
    location: dict  # GeoJSON Point
    footprint: Optional[dict] = None  # GeoJSON Polygon
    attributes: Optional[dict] = None
    verified: bool = False

    class Config:
        from_attributes = True


class DetectionResponse(BaseModel):
    """Response with all detections for a scene"""
    scene_id: UUID
    total_detections: int
    detections: List[DetectionResult]
    processing_time_ms: Optional[float] = None


class DetectionFilterRequest(BaseModel):
    """Filter parameters for querying detections"""
    aoi: Optional[GeoJSONPolygon] = None
    scene_ids: Optional[List[UUID]] = None
    detection_types: Optional[List[str]] = None
    min_confidence: float = 0.0
    verified_only: bool = False
    limit: int = 100


@router.post("/run", response_model=dict)
async def run_detection(
    request: DetectionRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session)
):
    """
    Run object detection on a satellite scene.
    Returns a task ID for tracking progress.
    """
    # Verify scene exists and is ready
    result = await session.execute(
        select(SatelliteScene).where(SatelliteScene.id == request.scene_id)
    )
    scene = result.scalar_one_or_none()

    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")

    if not scene.is_cog_ready:
        raise HTTPException(status_code=400, detail="Scene not ready for detection")

    # Queue detection job
    from services.detection import enqueue_detection
    import uuid

    task_id = str(uuid.uuid4())

    background_tasks.add_task(
        enqueue_detection,
        task_id=task_id,
        scene_id=request.scene_id,
        cog_url=scene.cog_url,
        aoi=request.aoi.model_dump() if request.aoi else None,
        detection_types=request.detection_types,
        confidence_threshold=request.confidence_threshold
    )

    return {
        "status": "queued",
        "task_id": task_id,
        "scene_id": str(request.scene_id),
        "message": "Detection job queued successfully"
    }


@router.get("/scene/{scene_id}", response_model=DetectionResponse)
async def get_scene_detections(
    scene_id: UUID,
    detection_type: Optional[str] = None,
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=500, ge=1, le=5000),
    session: AsyncSession = Depends(get_session)
):
    """Get all detections for a specific scene"""
    query = select(ObjectDetection).where(ObjectDetection.scene_id == scene_id)

    if detection_type:
        query = query.where(ObjectDetection.detection_type == detection_type)

    if min_confidence > 0:
        query = query.where(ObjectDetection.confidence >= min_confidence)

    query = query.order_by(ObjectDetection.confidence.desc()).limit(limit)

    result = await session.execute(query)
    detections = result.scalars().all()

    # Convert to GeoJSON
    detection_results = []
    for d in detections:
        location_geojson = await session.execute(
            select(ST_AsGeoJSON(d.location))
        )
        location = location_geojson.scalar_one()

        footprint = None
        if d.footprint:
            footprint_geojson = await session.execute(
                select(ST_AsGeoJSON(d.footprint))
            )
            footprint = footprint_geojson.scalar_one()

        detection_results.append(DetectionResult(
            id=d.id,
            detection_type=d.detection_type,
            class_name=d.class_name,
            confidence=float(d.confidence),
            location=eval(location),  # Parse JSON string
            footprint=eval(footprint) if footprint else None,
            attributes=d.attributes,
            verified=d.verified
        ))

    return DetectionResponse(
        scene_id=scene_id,
        total_detections=len(detection_results),
        detections=detection_results
    )


@router.post("/search")
async def search_detections(
    request: DetectionFilterRequest,
    session: AsyncSession = Depends(get_session)
):
    """Search detections with filters"""
    import json

    query = select(ObjectDetection)

    if request.aoi:
        aoi_geojson = json.dumps(request.aoi.model_dump())
        query = query.where(
            ST_Within(ObjectDetection.location, ST_GeomFromGeoJSON(aoi_geojson))
        )

    if request.scene_ids:
        query = query.where(ObjectDetection.scene_id.in_(request.scene_ids))

    if request.detection_types:
        query = query.where(ObjectDetection.detection_type.in_(request.detection_types))

    if request.min_confidence > 0:
        query = query.where(ObjectDetection.confidence >= request.min_confidence)

    if request.verified_only:
        query = query.where(ObjectDetection.verified == True)

    query = query.order_by(ObjectDetection.confidence.desc()).limit(request.limit)

    result = await session.execute(query)
    detections = result.scalars().all()

    # Build GeoJSON FeatureCollection
    features = []
    for d in detections:
        location_geojson = await session.execute(select(ST_AsGeoJSON(d.location)))
        location = eval(location_geojson.scalar_one())

        features.append({
            "type": "Feature",
            "geometry": location,
            "properties": {
                "id": d.id,
                "detection_type": d.detection_type,
                "class_name": d.class_name,
                "confidence": float(d.confidence),
                "scene_id": str(d.scene_id),
                "verified": d.verified,
                "detected_at": d.detected_at.isoformat() if d.detected_at else None,
                "attributes": d.attributes
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "total": len(features)
    }


@router.patch("/{detection_id}/verify")
async def verify_detection(
    detection_id: int,
    verified: bool,
    user_id: int = Query(..., description="User ID performing verification"),
    session: AsyncSession = Depends(get_session)
):
    """Mark a detection as verified or rejected"""
    result = await session.execute(
        select(ObjectDetection).where(ObjectDetection.id == detection_id)
    )
    detection = result.scalar_one_or_none()

    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    detection.verified = verified
    detection.verified_by = user_id
    detection.verified_at = datetime.utcnow()

    await session.commit()

    return {
        "id": detection_id,
        "verified": verified,
        "verified_by": user_id,
        "verified_at": detection.verified_at.isoformat()
    }


@router.get("/types")
async def list_detection_types():
    """List all available detection types and their models"""
    return {
        "types": [
            {
                "type": "vehicle",
                "classes": ["car", "truck", "bus", "tank", "apc"],
                "model": "yolov8_satellite_vehicles",
                "description": "Ground vehicles detection"
            },
            {
                "type": "ship",
                "classes": ["cargo", "tanker", "container", "military", "fishing"],
                "model": "yolov8_satellite_ships",
                "description": "Maritime vessel detection"
            },
            {
                "type": "aircraft",
                "classes": ["commercial", "military_jet", "helicopter", "drone"],
                "model": "yolov8_satellite_aircraft",
                "description": "Aircraft on ground or in air"
            },
            {
                "type": "building",
                "classes": ["residential", "commercial", "industrial", "military"],
                "model": "yolov8_satellite_buildings",
                "description": "Structure detection and classification"
            },
            {
                "type": "infrastructure",
                "classes": ["road", "bridge", "runway", "port", "solar_farm"],
                "model": "yolov8_satellite_infra",
                "description": "Infrastructure elements"
            }
        ]
    }


@router.get("/statistics")
async def get_detection_statistics(
    scene_id: Optional[UUID] = None,
    session: AsyncSession = Depends(get_session)
):
    """Get detection statistics"""
    query = select(
        ObjectDetection.detection_type,
        func.count(ObjectDetection.id).label("count"),
        func.avg(ObjectDetection.confidence).label("avg_confidence")
    ).group_by(ObjectDetection.detection_type)

    if scene_id:
        query = query.where(ObjectDetection.scene_id == scene_id)

    result = await session.execute(query)
    stats = result.all()

    return {
        "scene_id": str(scene_id) if scene_id else "all",
        "statistics": [
            {
                "detection_type": row.detection_type,
                "count": row.count,
                "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0
            }
            for row in stats
        ]
    }

"""
Areas of Interest (AOI) Router - CRUD operations for AOIs
"""
from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, delete, update, func
from sqlalchemy.ext.asyncio import AsyncSession
from geoalchemy2.functions import ST_AsGeoJSON, ST_GeomFromGeoJSON, ST_Area, ST_Transform

from database import get_db
from models.aoi import AreaOfInterest

router = APIRouter(prefix="/aois", tags=["Areas of Interest"])


# ===========================================
# Pydantic Models
# ===========================================

class GeoJSONGeometry(BaseModel):
    """GeoJSON Geometry"""
    type: str
    coordinates: list


class AOIStyle(BaseModel):
    """AOI display style"""
    color: str = "#3b82f6"
    fillOpacity: float = 0.3
    strokeWidth: int = 2


class AOIBase(BaseModel):
    """Base AOI data"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    geometry: GeoJSONGeometry
    style: Optional[AOIStyle] = None
    preferred_providers: Optional[List[str]] = None
    max_cloud_cover: int = Field(default=30, ge=0, le=100)


class AOICreate(AOIBase):
    """Create AOI request"""
    case_id: Optional[int] = None


class AOIUpdate(BaseModel):
    """Update AOI request"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    geometry: Optional[GeoJSONGeometry] = None
    style: Optional[AOIStyle] = None
    preferred_providers: Optional[List[str]] = None
    max_cloud_cover: Optional[int] = Field(None, ge=0, le=100)


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    enabled: bool
    interval_hours: int = Field(default=168, ge=1)  # Default weekly
    alert_on_change: bool = True
    alert_threshold: float = Field(default=0.1, ge=0, le=1)
    alert_email: Optional[str] = None


class AOIResponse(BaseModel):
    """AOI response with computed fields"""
    id: int
    user_id: int
    team_id: Optional[int]
    name: str
    description: Optional[str]
    geometry: dict
    style: Optional[dict]
    area_km2: Optional[float]

    # Monitoring
    monitor_enabled: bool
    monitor_interval_hours: int
    last_checked: Optional[datetime]
    next_check: Optional[datetime]
    alert_on_change: bool
    alert_threshold: float
    alert_email: Optional[str]

    # Preferences
    preferred_providers: Optional[List[str]]
    max_cloud_cover: int

    # Metadata
    case_id: Optional[int]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class AOIListResponse(BaseModel):
    """List of AOIs response"""
    aois: List[AOIResponse]
    total: int


# ===========================================
# Helper Functions
# ===========================================

def geometry_to_geojson(geometry) -> dict:
    """Convert PostGIS geometry to GeoJSON dict"""
    import json
    return json.loads(geometry)


# ===========================================
# Endpoints
# ===========================================

@router.get("", response_model=AOIListResponse)
async def list_aois(
    user_id: int = Query(..., description="User ID"),
    team_id: Optional[int] = Query(None, description="Filter by team ID"),
    monitor_enabled: Optional[bool] = Query(None, description="Filter by monitoring status"),
    case_id: Optional[int] = Query(None, description="Filter by case ID"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """
    List all AOIs for a user.
    """
    # Build query
    query = select(
        AreaOfInterest,
        ST_AsGeoJSON(AreaOfInterest.geometry).label("geojson"),
        ST_Area(ST_Transform(AreaOfInterest.geometry, 3857)).label("area_m2")
    ).where(AreaOfInterest.user_id == user_id)

    # Apply filters
    if team_id is not None:
        query = query.where(AreaOfInterest.team_id == team_id)
    if monitor_enabled is not None:
        query = query.where(AreaOfInterest.monitor_enabled == monitor_enabled)
    if case_id is not None:
        query = query.where(AreaOfInterest.case_id == case_id)

    # Count total
    count_query = select(func.count()).select_from(
        query.subquery()
    )
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination and order
    query = query.order_by(AreaOfInterest.updated_at.desc().nullslast())
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    rows = result.all()

    # Build response
    aois = []
    for row in rows:
        aoi = row[0]
        geojson = row[1]
        area_m2 = row[2]

        aois.append(AOIResponse(
            id=aoi.id,
            user_id=aoi.user_id,
            team_id=aoi.team_id,
            name=aoi.name,
            description=aoi.description,
            geometry=geometry_to_geojson(geojson) if geojson else {},
            style=aoi.style,
            area_km2=(area_m2 / 1_000_000) if area_m2 else None,
            monitor_enabled=aoi.monitor_enabled,
            monitor_interval_hours=aoi.monitor_interval_hours,
            last_checked=aoi.last_checked,
            next_check=aoi.next_check,
            alert_on_change=aoi.alert_on_change,
            alert_threshold=aoi.alert_threshold,
            alert_email=aoi.alert_email,
            preferred_providers=aoi.preferred_providers,
            max_cloud_cover=aoi.max_cloud_cover,
            case_id=aoi.case_id,
            created_at=aoi.created_at,
            updated_at=aoi.updated_at
        ))

    return AOIListResponse(aois=aois, total=total)


@router.post("", response_model=AOIResponse)
async def create_aoi(
    aoi_data: AOICreate,
    user_id: int = Query(..., description="User ID"),
    team_id: Optional[int] = Query(None, description="Team ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new Area of Interest.
    """
    import json

    # Convert GeoJSON to PostGIS geometry
    geojson_str = json.dumps({
        "type": aoi_data.geometry.type,
        "coordinates": aoi_data.geometry.coordinates
    })

    aoi = AreaOfInterest(
        user_id=user_id,
        team_id=team_id,
        name=aoi_data.name,
        description=aoi_data.description,
        geometry=ST_GeomFromGeoJSON(geojson_str),
        style=aoi_data.style.dict() if aoi_data.style else None,
        preferred_providers=aoi_data.preferred_providers,
        max_cloud_cover=aoi_data.max_cloud_cover,
        case_id=aoi_data.case_id,
        created_at=datetime.utcnow()
    )

    db.add(aoi)
    await db.commit()
    await db.refresh(aoi)

    # Fetch with computed fields
    result = await db.execute(
        select(
            AreaOfInterest,
            ST_AsGeoJSON(AreaOfInterest.geometry).label("geojson"),
            ST_Area(ST_Transform(AreaOfInterest.geometry, 3857)).label("area_m2")
        ).where(AreaOfInterest.id == aoi.id)
    )
    row = result.one()

    return AOIResponse(
        id=row[0].id,
        user_id=row[0].user_id,
        team_id=row[0].team_id,
        name=row[0].name,
        description=row[0].description,
        geometry=geometry_to_geojson(row[1]) if row[1] else {},
        style=row[0].style,
        area_km2=(row[2] / 1_000_000) if row[2] else None,
        monitor_enabled=row[0].monitor_enabled,
        monitor_interval_hours=row[0].monitor_interval_hours,
        last_checked=row[0].last_checked,
        next_check=row[0].next_check,
        alert_on_change=row[0].alert_on_change,
        alert_threshold=row[0].alert_threshold,
        alert_email=row[0].alert_email,
        preferred_providers=row[0].preferred_providers,
        max_cloud_cover=row[0].max_cloud_cover,
        case_id=row[0].case_id,
        created_at=row[0].created_at,
        updated_at=row[0].updated_at
    )


@router.get("/{aoi_id}", response_model=AOIResponse)
async def get_aoi(
    aoi_id: int,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific AOI by ID.
    """
    result = await db.execute(
        select(
            AreaOfInterest,
            ST_AsGeoJSON(AreaOfInterest.geometry).label("geojson"),
            ST_Area(ST_Transform(AreaOfInterest.geometry, 3857)).label("area_m2")
        ).where(
            AreaOfInterest.id == aoi_id,
            AreaOfInterest.user_id == user_id
        )
    )
    row = result.one_or_none()

    if not row:
        raise HTTPException(status_code=404, detail="AOI not found")

    return AOIResponse(
        id=row[0].id,
        user_id=row[0].user_id,
        team_id=row[0].team_id,
        name=row[0].name,
        description=row[0].description,
        geometry=geometry_to_geojson(row[1]) if row[1] else {},
        style=row[0].style,
        area_km2=(row[2] / 1_000_000) if row[2] else None,
        monitor_enabled=row[0].monitor_enabled,
        monitor_interval_hours=row[0].monitor_interval_hours,
        last_checked=row[0].last_checked,
        next_check=row[0].next_check,
        alert_on_change=row[0].alert_on_change,
        alert_threshold=row[0].alert_threshold,
        alert_email=row[0].alert_email,
        preferred_providers=row[0].preferred_providers,
        max_cloud_cover=row[0].max_cloud_cover,
        case_id=row[0].case_id,
        created_at=row[0].created_at,
        updated_at=row[0].updated_at
    )


@router.put("/{aoi_id}", response_model=AOIResponse)
async def update_aoi(
    aoi_id: int,
    aoi_data: AOIUpdate,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing AOI.
    """
    import json

    # Check ownership
    result = await db.execute(
        select(AreaOfInterest).where(
            AreaOfInterest.id == aoi_id,
            AreaOfInterest.user_id == user_id
        )
    )
    aoi = result.scalar_one_or_none()

    if not aoi:
        raise HTTPException(status_code=404, detail="AOI not found")

    # Update fields
    update_data = aoi_data.dict(exclude_unset=True)

    if "geometry" in update_data and update_data["geometry"]:
        geojson_str = json.dumps({
            "type": update_data["geometry"]["type"],
            "coordinates": update_data["geometry"]["coordinates"]
        })
        aoi.geometry = ST_GeomFromGeoJSON(geojson_str)
        del update_data["geometry"]

    if "style" in update_data and update_data["style"]:
        aoi.style = update_data["style"]
        del update_data["style"]

    for key, value in update_data.items():
        if hasattr(aoi, key):
            setattr(aoi, key, value)

    aoi.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(aoi)

    # Fetch with computed fields
    result = await db.execute(
        select(
            AreaOfInterest,
            ST_AsGeoJSON(AreaOfInterest.geometry).label("geojson"),
            ST_Area(ST_Transform(AreaOfInterest.geometry, 3857)).label("area_m2")
        ).where(AreaOfInterest.id == aoi.id)
    )
    row = result.one()

    return AOIResponse(
        id=row[0].id,
        user_id=row[0].user_id,
        team_id=row[0].team_id,
        name=row[0].name,
        description=row[0].description,
        geometry=geometry_to_geojson(row[1]) if row[1] else {},
        style=row[0].style,
        area_km2=(row[2] / 1_000_000) if row[2] else None,
        monitor_enabled=row[0].monitor_enabled,
        monitor_interval_hours=row[0].monitor_interval_hours,
        last_checked=row[0].last_checked,
        next_check=row[0].next_check,
        alert_on_change=row[0].alert_on_change,
        alert_threshold=row[0].alert_threshold,
        alert_email=row[0].alert_email,
        preferred_providers=row[0].preferred_providers,
        max_cloud_cover=row[0].max_cloud_cover,
        case_id=row[0].case_id,
        created_at=row[0].created_at,
        updated_at=row[0].updated_at
    )


@router.delete("/{aoi_id}")
async def delete_aoi(
    aoi_id: int,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete an AOI.
    """
    result = await db.execute(
        delete(AreaOfInterest).where(
            AreaOfInterest.id == aoi_id,
            AreaOfInterest.user_id == user_id
        ).returning(AreaOfInterest.id)
    )
    deleted = result.scalar_one_or_none()

    if not deleted:
        raise HTTPException(status_code=404, detail="AOI not found")

    await db.commit()

    return {"status": "deleted", "id": aoi_id}


@router.post("/{aoi_id}/monitoring", response_model=AOIResponse)
async def configure_monitoring(
    aoi_id: int,
    config: MonitoringConfig,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Configure monitoring for an AOI.
    """
    from datetime import timedelta

    # Check ownership
    result = await db.execute(
        select(AreaOfInterest).where(
            AreaOfInterest.id == aoi_id,
            AreaOfInterest.user_id == user_id
        )
    )
    aoi = result.scalar_one_or_none()

    if not aoi:
        raise HTTPException(status_code=404, detail="AOI not found")

    # Update monitoring config
    aoi.monitor_enabled = config.enabled
    aoi.monitor_interval_hours = config.interval_hours
    aoi.alert_on_change = config.alert_on_change
    aoi.alert_threshold = config.alert_threshold
    aoi.alert_email = config.alert_email

    if config.enabled:
        aoi.next_check = datetime.utcnow() + timedelta(hours=config.interval_hours)
    else:
        aoi.next_check = None

    aoi.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(aoi)

    # Fetch with computed fields
    result = await db.execute(
        select(
            AreaOfInterest,
            ST_AsGeoJSON(AreaOfInterest.geometry).label("geojson"),
            ST_Area(ST_Transform(AreaOfInterest.geometry, 3857)).label("area_m2")
        ).where(AreaOfInterest.id == aoi.id)
    )
    row = result.one()

    return AOIResponse(
        id=row[0].id,
        user_id=row[0].user_id,
        team_id=row[0].team_id,
        name=row[0].name,
        description=row[0].description,
        geometry=geometry_to_geojson(row[1]) if row[1] else {},
        style=row[0].style,
        area_km2=(row[2] / 1_000_000) if row[2] else None,
        monitor_enabled=row[0].monitor_enabled,
        monitor_interval_hours=row[0].monitor_interval_hours,
        last_checked=row[0].last_checked,
        next_check=row[0].next_check,
        alert_on_change=row[0].alert_on_change,
        alert_threshold=row[0].alert_threshold,
        alert_email=row[0].alert_email,
        preferred_providers=row[0].preferred_providers,
        max_cloud_cover=row[0].max_cloud_cover,
        case_id=row[0].case_id,
        created_at=row[0].created_at,
        updated_at=row[0].updated_at
    )


@router.post("/{aoi_id}/duplicate", response_model=AOIResponse)
async def duplicate_aoi(
    aoi_id: int,
    user_id: int = Query(..., description="User ID"),
    new_name: Optional[str] = Query(None, description="Name for duplicated AOI"),
    db: AsyncSession = Depends(get_db)
):
    """
    Duplicate an existing AOI.
    """
    # Get original
    result = await db.execute(
        select(
            AreaOfInterest,
            ST_AsGeoJSON(AreaOfInterest.geometry).label("geojson")
        ).where(
            AreaOfInterest.id == aoi_id,
            AreaOfInterest.user_id == user_id
        )
    )
    row = result.one_or_none()

    if not row:
        raise HTTPException(status_code=404, detail="AOI not found")

    original = row[0]

    # Create duplicate
    duplicate = AreaOfInterest(
        user_id=user_id,
        team_id=original.team_id,
        name=new_name or f"{original.name} (Copy)",
        description=original.description,
        geometry=original.geometry,
        style=original.style,
        preferred_providers=original.preferred_providers,
        max_cloud_cover=original.max_cloud_cover,
        monitor_enabled=False,  # Don't duplicate monitoring
        created_at=datetime.utcnow()
    )

    db.add(duplicate)
    await db.commit()
    await db.refresh(duplicate)

    # Fetch with computed fields
    result = await db.execute(
        select(
            AreaOfInterest,
            ST_AsGeoJSON(AreaOfInterest.geometry).label("geojson"),
            ST_Area(ST_Transform(AreaOfInterest.geometry, 3857)).label("area_m2")
        ).where(AreaOfInterest.id == duplicate.id)
    )
    row = result.one()

    return AOIResponse(
        id=row[0].id,
        user_id=row[0].user_id,
        team_id=row[0].team_id,
        name=row[0].name,
        description=row[0].description,
        geometry=geometry_to_geojson(row[1]) if row[1] else {},
        style=row[0].style,
        area_km2=(row[2] / 1_000_000) if row[2] else None,
        monitor_enabled=row[0].monitor_enabled,
        monitor_interval_hours=row[0].monitor_interval_hours,
        last_checked=row[0].last_checked,
        next_check=row[0].next_check,
        alert_on_change=row[0].alert_on_change,
        alert_threshold=row[0].alert_threshold,
        alert_email=row[0].alert_email,
        preferred_providers=row[0].preferred_providers,
        max_cloud_cover=row[0].max_cloud_cover,
        case_id=row[0].case_id,
        created_at=row[0].created_at,
        updated_at=row[0].updated_at
    )

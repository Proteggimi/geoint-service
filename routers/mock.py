"""
Mock Router - Provides mock data for testing without real providers
Used in lite mode for Render.com deployment
"""
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4
import random

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

router = APIRouter()


class MockScene(BaseModel):
    """Mock satellite scene"""
    id: str
    provider: str
    scene_id: str
    acquisition_date: datetime
    cloud_cover: float
    resolution_m: float
    sensor: str
    thumbnail_url: str
    bounds: List[float]  # [west, south, east, north]
    is_downloaded: bool = False
    is_cog_ready: bool = False


class MockSearchResponse(BaseModel):
    """Mock search response"""
    total: int
    scenes: List[MockScene]


class MockDetection(BaseModel):
    """Mock object detection"""
    id: str
    scene_id: str
    class_name: str
    confidence: float
    bbox: List[float]  # [x, y, width, height]
    centroid: List[float]  # [lat, lon]


# Sample locations for mock data (European locations of interest)
MOCK_LOCATIONS = [
    {"name": "Sevastopol", "lat": 44.6054, "lon": 33.5220, "bounds": [33.4, 44.5, 33.7, 44.7]},
    {"name": "Mariupol", "lat": 47.0970, "lon": 37.5430, "bounds": [37.4, 47.0, 37.7, 47.2]},
    {"name": "Kherson", "lat": 46.6354, "lon": 32.6169, "bounds": [32.5, 46.5, 32.8, 46.7]},
    {"name": "Odessa Port", "lat": 46.4846, "lon": 30.7326, "bounds": [30.6, 46.4, 30.9, 46.6]},
    {"name": "Kaliningrad", "lat": 54.7104, "lon": 20.4522, "bounds": [20.3, 54.6, 20.6, 54.8]},
]

MOCK_SENSORS = [
    {"provider": "sentinel", "sensor": "S2A-MSI", "resolution": 10},
    {"provider": "sentinel", "sensor": "S2B-MSI", "resolution": 10},
    {"provider": "sentinel", "sensor": "S1A-SAR", "resolution": 5},
    {"provider": "planet", "sensor": "PSScene", "resolution": 3},
    {"provider": "maxar", "sensor": "WV03", "resolution": 0.3},
]

DETECTION_CLASSES = [
    "vehicle", "ship", "aircraft", "building", "tank",
    "military_vehicle", "cargo_ship", "container"
]


def generate_mock_scenes(
    lat: float,
    lon: float,
    count: int = 10,
    start_date: datetime = None,
    end_date: datetime = None
) -> List[MockScene]:
    """Generate mock satellite scenes for a location"""
    scenes = []

    if not start_date:
        start_date = datetime.now() - timedelta(days=90)
    if not end_date:
        end_date = datetime.now()

    date_range = (end_date - start_date).days

    for i in range(count):
        sensor = random.choice(MOCK_SENSORS)
        days_offset = random.randint(0, date_range)
        acq_date = start_date + timedelta(days=days_offset, hours=random.randint(8, 16))

        # Generate bounds around the point
        offset = 0.1 + random.random() * 0.2
        bounds = [
            lon - offset,
            lat - offset,
            lon + offset,
            lat + offset
        ]

        scene = MockScene(
            id=str(uuid4()),
            provider=sensor["provider"],
            scene_id=f"{sensor['sensor']}_{acq_date.strftime('%Y%m%d')}_{i:04d}",
            acquisition_date=acq_date,
            cloud_cover=random.uniform(0, 30),
            resolution_m=sensor["resolution"],
            sensor=sensor["sensor"],
            thumbnail_url=f"https://placehold.co/256x256/2d3748/e2e8f0?text={sensor['sensor']}",
            bounds=bounds,
            is_downloaded=random.choice([True, False]),
            is_cog_ready=random.choice([True, False])
        )
        scenes.append(scene)

    # Sort by date descending
    scenes.sort(key=lambda x: x.acquisition_date, reverse=True)
    return scenes


def generate_mock_detections(scene_id: str, count: int = 5) -> List[MockDetection]:
    """Generate mock object detections for a scene"""
    detections = []

    for i in range(count):
        # Random position within image (normalized 0-1)
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        w = random.uniform(0.02, 0.08)
        h = random.uniform(0.02, 0.08)

        # Generate fake lat/lon
        lat = 44.0 + random.uniform(-0.5, 0.5)
        lon = 33.0 + random.uniform(-0.5, 0.5)

        detection = MockDetection(
            id=str(uuid4()),
            scene_id=scene_id,
            class_name=random.choice(DETECTION_CLASSES),
            confidence=random.uniform(0.65, 0.98),
            bbox=[x, y, w, h],
            centroid=[lat, lon]
        )
        detections.append(detection)

    return detections


# Mock endpoints
@router.get("/scenes", response_model=MockSearchResponse)
async def mock_search_scenes(
    lat: float = Query(44.6, description="Center latitude"),
    lon: float = Query(33.5, description="Center longitude"),
    days: int = Query(30, description="Days of history"),
    limit: int = Query(20, ge=1, le=100),
    provider: Optional[str] = Query(None, description="Filter by provider")
):
    """
    Mock scene search endpoint.
    Returns simulated satellite scenes for testing the frontend.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    scenes = generate_mock_scenes(lat, lon, count=limit * 2, start_date=start_date, end_date=end_date)

    if provider:
        scenes = [s for s in scenes if s.provider == provider]

    scenes = scenes[:limit]

    return MockSearchResponse(total=len(scenes), scenes=scenes)


@router.get("/scenes/{scene_id}")
async def mock_get_scene(scene_id: str):
    """Get a mock scene by ID"""
    location = random.choice(MOCK_LOCATIONS)
    sensor = random.choice(MOCK_SENSORS)

    return MockScene(
        id=scene_id,
        provider=sensor["provider"],
        scene_id=f"MOCK_{scene_id[:8]}",
        acquisition_date=datetime.now() - timedelta(days=random.randint(1, 30)),
        cloud_cover=random.uniform(0, 20),
        resolution_m=sensor["resolution"],
        sensor=sensor["sensor"],
        thumbnail_url=f"https://placehold.co/512x512/2d3748/e2e8f0?text={sensor['sensor']}",
        bounds=location["bounds"],
        is_downloaded=True,
        is_cog_ready=True
    )


@router.post("/search", response_model=MockSearchResponse)
async def mock_search_scenes_post(
    aoi: dict,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    providers: Optional[List[str]] = None,
    max_cloud_cover: float = 30.0,
    limit: int = 50
):
    """
    Mock scene search with AOI polygon.
    Accepts same format as real /scenes/search endpoint.
    Works when mounted at /scenes prefix -> /scenes/search
    """
    # Extract center from AOI polygon
    if "coordinates" in aoi:
        coords = aoi["coordinates"][0]  # First ring of polygon
        lats = [c[1] for c in coords]
        lons = [c[0] for c in coords]
        lat = sum(lats) / len(lats)
        lon = sum(lons) / len(lons)
    else:
        lat, lon = 44.6, 33.5  # Default

    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    scenes = generate_mock_scenes(lat, lon, count=limit, start_date=start_date, end_date=end_date)

    # Filter by cloud cover
    scenes = [s for s in scenes if s.cloud_cover <= max_cloud_cover]

    # Filter by providers
    if providers:
        scenes = [s for s in scenes if s.provider in providers]

    return MockSearchResponse(total=len(scenes), scenes=scenes)


@router.get("/detections/{scene_id}")
async def mock_get_detections(
    scene_id: str,
    min_confidence: float = Query(0.5, ge=0, le=1)
):
    """Get mock object detections for a scene"""
    detections = generate_mock_detections(scene_id, count=random.randint(3, 15))
    detections = [d for d in detections if d.confidence >= min_confidence]

    return {
        "scene_id": scene_id,
        "total": len(detections),
        "detections": detections
    }


@router.get("/analysis/change-detection")
async def mock_change_detection(
    scene_before: str = Query(..., description="Before scene ID"),
    scene_after: str = Query(..., description="After scene ID")
):
    """Mock change detection between two scenes"""
    # Generate random change areas
    changes = []
    for i in range(random.randint(2, 8)):
        changes.append({
            "id": str(uuid4()),
            "type": random.choice(["new_construction", "destruction", "vegetation_loss", "vehicle_movement"]),
            "confidence": random.uniform(0.7, 0.95),
            "area_sqm": random.uniform(100, 5000),
            "centroid": [44.6 + random.uniform(-0.2, 0.2), 33.5 + random.uniform(-0.2, 0.2)],
            "severity": random.choice(["low", "medium", "high"])
        })

    return {
        "scene_before": scene_before,
        "scene_after": scene_after,
        "analysis_date": datetime.now().isoformat(),
        "total_changes": len(changes),
        "changes": changes,
        "summary": {
            "new_construction": len([c for c in changes if c["type"] == "new_construction"]),
            "destruction": len([c for c in changes if c["type"] == "destruction"]),
            "vegetation_loss": len([c for c in changes if c["type"] == "vegetation_loss"]),
            "vehicle_movement": len([c for c in changes if c["type"] == "vehicle_movement"])
        }
    }


@router.get("/tiles/{scene_id}/{z}/{x}/{y}.png")
async def mock_tile(scene_id: str, z: int, x: int, y: int):
    """Mock tile endpoint - returns placeholder image URL"""
    from fastapi.responses import RedirectResponse
    # Return a placeholder tile image
    return RedirectResponse(
        url=f"https://placehold.co/256x256/1a365d/e2e8f0?text={z}/{x}/{y}"
    )


@router.get("/locations")
async def get_mock_locations():
    """Get predefined mock locations for testing"""
    return {
        "locations": MOCK_LOCATIONS,
        "description": "Predefined locations for GEOINT testing"
    }

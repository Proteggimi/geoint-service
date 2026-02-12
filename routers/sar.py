"""
SAR Processing Router - Endpoints for SAR data processing
"""
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

from services.sar import (
    sar_processor,
    Polarization,
    SpeckleFilter,
    Calibration
)

router = APIRouter(prefix="/sar", tags=["SAR Processing"])
logger = structlog.get_logger()


# ===========================================
# Pydantic Models
# ===========================================

class SARProcessingRequest(BaseModel):
    """Request for SAR processing"""
    scene_id: str
    polarization: str = Field(default="VV", description="Polarization: VV, VH, VV+VH, VV/VH")
    speckle_filter: str = Field(default="lee", description="Filter: none, lee, frost, gamma_map, refined_lee")
    filter_window_size: int = Field(default=5, ge=3, le=15)
    calibration: str = Field(default="sigma0", description="Calibration: sigma0, gamma0, beta0")
    terrain_correction: bool = Field(default=True)
    min_db: float = Field(default=-25)
    max_db: float = Field(default=0)
    colormap: str = Field(default="grayscale")


class SARProcessingResponse(BaseModel):
    """Response from SAR processing"""
    scene_id: str
    processed_url: str
    thumbnail_url: Optional[str]
    statistics: dict
    processing_params: dict


class CoherenceRequest(BaseModel):
    """Request for coherence computation"""
    scene_id_1: str
    scene_id_2: str
    window_size: int = Field(default=5, ge=3, le=15)


class CoherenceResponse(BaseModel):
    """Response from coherence computation"""
    scene_id_1: str
    scene_id_2: str
    coherence_url: str
    mean_coherence: float
    statistics: dict


class PolarimetricRequest(BaseModel):
    """Request for polarimetric decomposition"""
    scene_id: str
    products: List[str] = Field(
        default=["ratio", "rvi"],
        description="Products: ratio, normalized_diff, rvi"
    )


class PolarimetricResponse(BaseModel):
    """Response from polarimetric decomposition"""
    scene_id: str
    products: dict  # product_name -> url


# ===========================================
# Endpoints
# ===========================================

@router.post("/process", response_model=SARProcessingResponse)
async def process_sar_scene(
    request: SARProcessingRequest,
    background_tasks: BackgroundTasks
):
    """
    Process a SAR scene with specified parameters.

    Applies calibration, terrain correction, and speckle filtering.
    """
    logger.info(
        "SAR processing request",
        scene_id=request.scene_id,
        polarization=request.polarization
    )

    try:
        # Validate enums
        polarization = Polarization(request.polarization)
        speckle_filter = SpeckleFilter(request.speckle_filter)
        calibration = Calibration(request.calibration)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {e}")

    # In production, this would:
    # 1. Load scene from storage
    # 2. Process with sar_processor
    # 3. Save result and return URL

    # Mock response for now
    return SARProcessingResponse(
        scene_id=request.scene_id,
        processed_url=f"/geoint/sar/tiles/{request.scene_id}/{{z}}/{{x}}/{{y}}.png",
        thumbnail_url=f"/geoint/sar/preview/{request.scene_id}.png",
        statistics={
            "min_db": request.min_db,
            "max_db": request.max_db,
            "mean_db": -12.5,
            "std_db": 4.2
        },
        processing_params={
            "polarization": request.polarization,
            "speckle_filter": request.speckle_filter,
            "calibration": request.calibration,
            "terrain_correction": request.terrain_correction
        }
    )


@router.post("/coherence", response_model=CoherenceResponse)
async def compute_coherence(request: CoherenceRequest):
    """
    Compute interferometric coherence between two SAR scenes.

    Coherence values range from 0 (no correlation) to 1 (perfect correlation).
    Low coherence indicates surface changes between acquisitions.
    """
    logger.info(
        "Coherence computation request",
        scene_1=request.scene_id_1,
        scene_2=request.scene_id_2
    )

    # In production, this would load both scenes and compute coherence
    # Mock response for now
    return CoherenceResponse(
        scene_id_1=request.scene_id_1,
        scene_id_2=request.scene_id_2,
        coherence_url=f"/geoint/sar/coherence/{request.scene_id_1}_{request.scene_id_2}.tif",
        mean_coherence=0.72,
        statistics={
            "min": 0.1,
            "max": 0.95,
            "mean": 0.72,
            "std": 0.18,
            "low_coherence_pct": 15.3,  # % of pixels with coherence < 0.3
            "high_coherence_pct": 45.2  # % of pixels with coherence > 0.7
        }
    )


@router.post("/polarimetric", response_model=PolarimetricResponse)
async def compute_polarimetric_products(request: PolarimetricRequest):
    """
    Compute polarimetric decomposition products.

    Available products:
    - ratio: VV/VH ratio
    - normalized_diff: (VV-VH)/(VV+VH)
    - rvi: Radar Vegetation Index
    """
    logger.info(
        "Polarimetric decomposition request",
        scene_id=request.scene_id,
        products=request.products
    )

    # Mock response
    products = {}
    for product in request.products:
        products[product] = f"/geoint/sar/polarimetric/{request.scene_id}/{product}.tif"

    return PolarimetricResponse(
        scene_id=request.scene_id,
        products=products
    )


@router.get("/filters")
async def list_speckle_filters():
    """
    List available speckle filters with descriptions.
    """
    return {
        "filters": [
            {
                "id": "none",
                "name": "None",
                "description": "No filtering applied"
            },
            {
                "id": "lee",
                "name": "Lee Filter",
                "description": "Adaptive filter based on local statistics. Good general-purpose filter.",
                "parameters": {"window_size": {"default": 5, "range": [3, 15]}}
            },
            {
                "id": "frost",
                "name": "Frost Filter",
                "description": "Exponentially weighted filter. Better edge preservation than Lee.",
                "parameters": {
                    "window_size": {"default": 5, "range": [3, 15]},
                    "damping": {"default": 2.0, "range": [0.5, 5.0]}
                }
            },
            {
                "id": "gamma_map",
                "name": "Gamma MAP Filter",
                "description": "Maximum a posteriori filter assuming Gamma distribution. Good for multi-look data.",
                "parameters": {
                    "window_size": {"default": 5, "range": [3, 15]},
                    "nlooks": {"default": 1, "range": [1, 16]}
                }
            },
            {
                "id": "refined_lee",
                "name": "Refined Lee Filter",
                "description": "Edge-preserving Lee filter using directional windows. Best for preserving linear features.",
                "parameters": {"window_size": {"default": 7, "range": [5, 15]}}
            }
        ]
    }


@router.get("/calibration")
async def list_calibration_options():
    """
    List available radiometric calibration options.
    """
    return {
        "calibrations": [
            {
                "id": "sigma0",
                "name": "Sigma Nought (σ°)",
                "description": "Radar cross section per unit ground area. Most common for land applications.",
                "unit": "m²/m²"
            },
            {
                "id": "gamma0",
                "name": "Gamma Nought (γ°)",
                "description": "Normalized to local incidence angle. Reduces topographic effects.",
                "unit": "m²/m²"
            },
            {
                "id": "beta0",
                "name": "Beta Nought (β°)",
                "description": "Radar brightness coefficient. Normalized to slant range.",
                "unit": "m²/m²"
            }
        ]
    }


@router.get("/colormaps")
async def list_sar_colormaps():
    """
    List available colormaps for SAR visualization.
    """
    return {
        "colormaps": [
            {
                "id": "grayscale",
                "name": "Grayscale",
                "description": "Traditional black to white gradient",
                "colors": ["#000000", "#ffffff"]
            },
            {
                "id": "viridis",
                "name": "Viridis",
                "description": "Perceptually uniform, colorblind-friendly",
                "colors": ["#440154", "#21918c", "#fde725"]
            },
            {
                "id": "plasma",
                "name": "Plasma",
                "description": "High contrast for detail visibility",
                "colors": ["#0d0887", "#cc4778", "#f0f921"]
            },
            {
                "id": "terrain",
                "name": "Terrain",
                "description": "Emphasizes topographic features",
                "colors": ["#333399", "#00cc00", "#ffff00", "#ff6600"]
            },
            {
                "id": "ocean",
                "name": "Ocean",
                "description": "Optimized for maritime applications",
                "colors": ["#000033", "#006699", "#00ccff"]
            }
        ]
    }


@router.get("/tiles/{scene_id}/{z}/{x}/{y}.png")
async def get_sar_tile(
    scene_id: str,
    z: int,
    x: int,
    y: int,
    polarization: str = Query("VV"),
    speckle_filter: str = Query("lee"),
    min_db: float = Query(-25),
    max_db: float = Query(0),
    colormap: str = Query("grayscale")
):
    """
    Get a processed SAR tile.

    Applies speckle filtering on-the-fly.
    """
    # This would generate tiles from processed SAR data
    # For now, return 404 as actual implementation requires data
    raise HTTPException(
        status_code=404,
        detail="Scene not found or not processed"
    )


@router.get("/preview/{scene_id}.png")
async def get_sar_preview(
    scene_id: str,
    width: int = Query(512, ge=64, le=2048),
    height: int = Query(512, ge=64, le=2048),
    polarization: str = Query("VV"),
    colormap: str = Query("grayscale")
):
    """
    Get a preview image of processed SAR scene.
    """
    raise HTTPException(
        status_code=404,
        detail="Scene not found or not processed"
    )

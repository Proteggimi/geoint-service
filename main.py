"""
GEOINT Service - Main Application
Microservizio FastAPI per analisi GEOINT e SAR
"""
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from config import get_settings
from database import init_db, close_db

settings = get_settings()

# Conditional imports based on lite_mode
if settings.lite_mode:
    # In lite mode, only import mock router
    from routers import mock
else:
    # Full mode with all routers
    from routers import scenes, tiles, analysis, detections, providers, aois, sar

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting GEOINT Service", version=settings.app_version)
    await init_db()
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Shutting down GEOINT Service")
    await close_db()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    GEOINT (Geospatial Intelligence) and SAR (Synthetic Aperture Radar) Analysis Service.

    Features:
    - Satellite imagery search and download (Sentinel, Planet, Maxar, Capella)
    - Cloud-Optimized GeoTIFF tile serving
    - Change detection analysis
    - Object detection (vehicles, ships, infrastructure)
    - Multi-band visualization (RGB, NIR, NDVI)
    """,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    logger.info(
        "Request",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None
    )
    response = await call_next(request)
    logger.info(
        "Response",
        method=request.method,
        path=request.url.path,
        status=response.status_code
    )
    return response


# Include routers based on mode
if settings.lite_mode:
    # Lite mode - only mock endpoints for testing
    app.include_router(mock.router, prefix="/mock", tags=["Mock Data"])
    logger.info("Running in LITE MODE - mock endpoints enabled")
else:
    # Full mode - all real endpoints
    app.include_router(scenes.router, prefix="/scenes", tags=["Scenes"])
    app.include_router(tiles.router, prefix="/tiles", tags=["Tiles"])
    app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
    app.include_router(detections.router, prefix="/detections", tags=["Object Detection"])
    app.include_router(providers.router, prefix="/providers", tags=["Providers"])
    app.include_router(aois.router, tags=["Areas of Interest"])
    app.include_router(sar.router, tags=["SAR Processing"])


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "version": settings.app_version}


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Readiness check - verifies all dependencies"""
    from database import check_db_connection

    checks = {
        "database": await check_db_connection(),
    }

    # Only check storage in full mode
    if not settings.lite_mode:
        from services.storage import check_storage_connection
        checks["storage"] = await check_storage_connection()
    else:
        checks["lite_mode"] = True

    all_healthy = all(v for k, v in checks.items() if k != "lite_mode")

    return {
        "status": "ready" if all_healthy else "not_ready",
        "mode": "lite" if settings.lite_mode else "full",
        "checks": checks
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info"""
    response = {
        "service": settings.app_name,
        "version": settings.app_version,
        "mode": "lite" if settings.lite_mode else "full",
        "docs": "/docs" if settings.debug else "disabled",
    }

    if settings.lite_mode:
        response["endpoints"] = {
            "mock_scenes": "/mock/scenes",
            "mock_search": "/mock/scenes/search",
            "mock_detections": "/mock/detections/{scene_id}",
            "mock_change_detection": "/mock/analysis/change-detection",
            "mock_locations": "/mock/locations"
        }
    else:
        response["providers"] = {
            "sentinel": settings.sentinel_hub_enabled,
            "planet": settings.planet_enabled,
            "maxar": settings.maxar_enabled,
            "capella": settings.capella_enabled,
        }

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
    )

"""
Providers Router - Manage satellite data providers
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from config import get_settings

router = APIRouter()
settings = get_settings()


class ProviderStatus(BaseModel):
    """Provider status and configuration"""
    name: str
    enabled: bool
    configured: bool
    tier: str
    description: str
    capabilities: list[str]


class ProviderConfigRequest(BaseModel):
    """Request to configure a provider"""
    api_key: Optional[str] = Field(default=None)
    client_id: Optional[str] = Field(default=None)
    client_secret: Optional[str] = Field(default=None)
    enabled: bool = True


@router.get("/", response_model=list[ProviderStatus])
async def list_providers():
    """List all available satellite data providers and their status"""
    providers = [
        ProviderStatus(
            name="sentinel",
            enabled=settings.sentinel_hub_enabled,
            configured=bool(settings.sentinel_hub_client_id and settings.sentinel_hub_client_secret),
            tier="free",
            description="ESA Copernicus Sentinel-1 (SAR) and Sentinel-2 (optical) imagery",
            capabilities=[
                "optical_10m",
                "sar_5m",
                "multispectral",
                "free_access",
                "global_coverage"
            ]
        ),
        ProviderStatus(
            name="planet",
            enabled=settings.planet_enabled,
            configured=bool(settings.planet_api_key),
            tier="basic",
            description="Planet Labs high-frequency 3-5m optical imagery",
            capabilities=[
                "optical_3m",
                "daily_revisit",
                "global_coverage",
                "api_access"
            ]
        ),
        ProviderStatus(
            name="maxar",
            enabled=settings.maxar_enabled,
            configured=bool(settings.maxar_api_key),
            tier="premium",
            description="Maxar WorldView 30cm resolution optical imagery",
            capabilities=[
                "optical_30cm",
                "highest_resolution",
                "stereo_imaging",
                "archive_access"
            ]
        ),
        ProviderStatus(
            name="capella",
            enabled=settings.capella_enabled,
            configured=bool(settings.capella_api_key),
            tier="premium",
            description="Capella Space SAR with 50cm resolution",
            capabilities=[
                "sar_50cm",
                "all_weather",
                "day_night",
                "rapid_tasking"
            ]
        )
    ]

    return providers


@router.get("/{provider_name}")
async def get_provider_details(provider_name: str):
    """Get detailed information about a specific provider"""
    provider_details = {
        "sentinel": {
            "name": "Copernicus Sentinel",
            "operator": "European Space Agency (ESA)",
            "website": "https://scihub.copernicus.eu/",
            "satellites": [
                {
                    "name": "Sentinel-1",
                    "type": "SAR",
                    "bands": ["VV", "VH"],
                    "resolution": "5m",
                    "revisit": "6 days",
                    "swath": "250km"
                },
                {
                    "name": "Sentinel-2",
                    "type": "Optical/Multispectral",
                    "bands": 13,
                    "resolution": "10m (visible), 20m (red edge), 60m (atmosphere)",
                    "revisit": "5 days",
                    "swath": "290km"
                }
            ],
            "pricing": {
                "model": "Free",
                "note": "Open access for all users worldwide"
            },
            "api_endpoints": {
                "catalog": "https://scihub.copernicus.eu/dhus/",
                "process": "https://services.sentinel-hub.com/"
            }
        },
        "planet": {
            "name": "Planet Labs",
            "operator": "Planet Labs PBC",
            "website": "https://www.planet.com/",
            "satellites": [
                {
                    "name": "PlanetScope",
                    "type": "Optical",
                    "bands": 4,
                    "resolution": "3-4m",
                    "revisit": "Daily",
                    "constellation": "~200 satellites"
                },
                {
                    "name": "SkySat",
                    "type": "Optical",
                    "bands": 4,
                    "resolution": "50cm",
                    "revisit": "4-5 times/day",
                    "video": True
                }
            ],
            "pricing": {
                "model": "Subscription + Per-Area",
                "tiers": ["Explorer", "Standard", "Premium"]
            }
        },
        "maxar": {
            "name": "Maxar Technologies",
            "operator": "Maxar Technologies",
            "website": "https://www.maxar.com/",
            "satellites": [
                {
                    "name": "WorldView-3",
                    "type": "Optical/Multispectral",
                    "bands": 29,
                    "resolution": "31cm",
                    "revisit": "< 1 day"
                },
                {
                    "name": "WorldView Legion",
                    "type": "Optical",
                    "bands": 5,
                    "resolution": "30cm",
                    "revisit": "15x daily",
                    "constellation": "6 satellites"
                }
            ],
            "pricing": {
                "model": "Per-Area License",
                "archive": "$15-25/km²",
                "tasking": "$25-35/km²"
            }
        },
        "capella": {
            "name": "Capella Space",
            "operator": "Capella Space Corp",
            "website": "https://www.capellaspace.com/",
            "satellites": [
                {
                    "name": "Capella SAR",
                    "type": "SAR (X-band)",
                    "resolution": "50cm (Spotlight)",
                    "modes": ["Spotlight", "Stripmap", "ScanSAR"],
                    "revisit": "< 6 hours",
                    "all_weather": True,
                    "day_night": True
                }
            ],
            "pricing": {
                "model": "Per-Collection",
                "archive": "$350-500/scene",
                "tasking": "$1500-5000/scene"
            }
        }
    }

    if provider_name not in provider_details:
        raise HTTPException(status_code=404, detail="Provider not found")

    return provider_details[provider_name]


@router.post("/{provider_name}/configure")
async def configure_provider(
    provider_name: str,
    config: ProviderConfigRequest
):
    """
    Configure API credentials for a provider.
    Note: In production, this would update secure storage.
    """
    valid_providers = ["sentinel", "planet", "maxar", "capella"]

    if provider_name not in valid_providers:
        raise HTTPException(status_code=404, detail="Provider not found")

    # In production, save to secure storage (e.g., HashiCorp Vault, AWS Secrets Manager)
    # For now, just validate the configuration

    if provider_name == "sentinel":
        if not config.client_id or not config.client_secret:
            raise HTTPException(
                status_code=400,
                detail="Sentinel Hub requires client_id and client_secret"
            )
    else:
        if not config.api_key:
            raise HTTPException(
                status_code=400,
                detail=f"{provider_name} requires api_key"
            )

    return {
        "provider": provider_name,
        "status": "configured",
        "enabled": config.enabled,
        "message": f"Provider {provider_name} configured successfully"
    }


@router.post("/{provider_name}/test")
async def test_provider_connection(provider_name: str):
    """Test connection to a provider's API"""
    from services.providers import get_provider_factory

    factory = get_provider_factory()
    provider = factory.get_provider(provider_name)

    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found or not configured")

    try:
        is_healthy = await provider.health_check()
        return {
            "provider": provider_name,
            "status": "healthy" if is_healthy else "unhealthy",
            "message": "Connection successful" if is_healthy else "Connection failed"
        }
    except Exception as e:
        return {
            "provider": provider_name,
            "status": "error",
            "message": str(e)
        }


@router.get("/{provider_name}/quota")
async def get_provider_quota(provider_name: str):
    """Get current usage and quota for a provider"""
    # Mock response - in production, query actual API
    quotas = {
        "sentinel": {
            "plan": "Free Tier",
            "requests_today": 150,
            "requests_limit": 3000,
            "data_downloaded_mb": 2500,
            "data_limit_mb": None,  # Unlimited
            "reset_at": "2024-01-16T00:00:00Z"
        },
        "planet": {
            "plan": "Explorer",
            "requests_today": 45,
            "requests_limit": 500,
            "area_downloaded_km2": 1200,
            "area_limit_km2": 5000,
            "reset_at": "2024-02-01T00:00:00Z"
        }
    }

    if provider_name not in quotas:
        raise HTTPException(status_code=404, detail="Provider not found or quota not available")

    return quotas[provider_name]

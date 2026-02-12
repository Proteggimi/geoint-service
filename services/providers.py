"""
Satellite Provider Factory - Modular provider management
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Dict, Any
from functools import lru_cache
import httpx
import structlog

from config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class SatelliteProvider(ABC):
    """Abstract base class for satellite data providers"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name"""
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if provider is enabled and configured"""
        pass

    @abstractmethod
    async def search(
        self,
        aoi: dict,
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: Optional[float] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search for scenes in the given AOI and date range"""
        pass

    @abstractmethod
    async def get_download_url(self, scene_id: str) -> str:
        """Get download URL for a scene"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider API is accessible"""
        pass


class SentinelProvider(SatelliteProvider):
    """Copernicus Sentinel Hub provider"""

    def __init__(self):
        self.client_id = settings.sentinel_hub_client_id
        self.client_secret = settings.sentinel_hub_client_secret
        self._token: Optional[str] = None
        self._token_expires: Optional[datetime] = None

    @property
    def name(self) -> str:
        return "sentinel"

    def is_enabled(self) -> bool:
        return settings.sentinel_hub_enabled

    async def _get_token(self) -> str:
        """Get OAuth token for Sentinel Hub"""
        if self._token and self._token_expires and datetime.utcnow() < self._token_expires:
            return self._token

        if not self.client_id or not self.client_secret:
            # Use free STAC API without auth
            return ""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://services.sentinel-hub.com/oauth/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret
                }
            )
            response.raise_for_status()
            data = response.json()
            self._token = data["access_token"]
            # Set expiry 5 minutes before actual expiry
            from datetime import timedelta
            self._token_expires = datetime.utcnow() + timedelta(seconds=data["expires_in"] - 300)
            return self._token

    async def search(
        self,
        aoi: dict,
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: Optional[float] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search Sentinel scenes using STAC API"""

        # Use Microsoft Planetary Computer STAC (free, no auth required)
        stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

        query = {
            "collections": ["sentinel-2-l2a"],
            "intersects": aoi,
            "datetime": f"{start_date.isoformat()}/{end_date.isoformat()}",
            "limit": limit,
            "sortby": [{"field": "datetime", "direction": "desc"}]
        }

        if max_cloud_cover is not None:
            query["query"] = {"eo:cloud_cover": {"lte": max_cloud_cover}}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(stac_url, json=query)
            response.raise_for_status()
            data = response.json()

        scenes = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            scenes.append({
                "provider": "sentinel",
                "scene_id": feature.get("id"),
                "acquisition_date": props.get("datetime"),
                "cloud_cover": props.get("eo:cloud_cover"),
                "resolution_m": 10,
                "sensor": props.get("platform"),
                "footprint": feature.get("geometry"),
                "thumbnail_url": feature.get("assets", {}).get("thumbnail", {}).get("href"),
                "metadata": props
            })

        logger.info("Sentinel search completed", count=len(scenes))
        return scenes

    async def get_download_url(self, scene_id: str) -> str:
        """Get download URL from STAC"""
        stac_url = f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/{scene_id}"

        async with httpx.AsyncClient() as client:
            response = await client.get(stac_url)
            response.raise_for_status()
            data = response.json()

        # Return visual asset URL (COG)
        return data.get("assets", {}).get("visual", {}).get("href", "")

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://planetarycomputer.microsoft.com/api/stac/v1"
                )
                return response.status_code == 200
        except Exception:
            return False


class PlanetProvider(SatelliteProvider):
    """Planet Labs provider"""

    def __init__(self):
        self.api_key = settings.planet_api_key

    @property
    def name(self) -> str:
        return "planet"

    def is_enabled(self) -> bool:
        return settings.planet_enabled and bool(self.api_key)

    async def search(
        self,
        aoi: dict,
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: Optional[float] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search Planet scenes"""
        if not self.is_enabled():
            return []

        # Planet Data API
        search_url = "https://api.planet.com/data/v1/quick-search"

        filters = {
            "type": "AndFilter",
            "config": [
                {
                    "type": "GeometryFilter",
                    "field_name": "geometry",
                    "config": aoi
                },
                {
                    "type": "DateRangeFilter",
                    "field_name": "acquired",
                    "config": {
                        "gte": start_date.isoformat() + "Z",
                        "lte": end_date.isoformat() + "Z"
                    }
                }
            ]
        }

        if max_cloud_cover is not None:
            filters["config"].append({
                "type": "RangeFilter",
                "field_name": "cloud_cover",
                "config": {"lte": max_cloud_cover / 100}  # Planet uses 0-1
            })

        request_body = {
            "item_types": ["PSScene"],
            "filter": filters
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                search_url,
                json=request_body,
                auth=(self.api_key, "")
            )
            response.raise_for_status()
            data = response.json()

        scenes = []
        for feature in data.get("features", [])[:limit]:
            props = feature.get("properties", {})
            scenes.append({
                "provider": "planet",
                "scene_id": feature.get("id"),
                "acquisition_date": props.get("acquired"),
                "cloud_cover": props.get("cloud_cover", 0) * 100,
                "resolution_m": props.get("gsd", 3),
                "sensor": props.get("instrument"),
                "footprint": feature.get("geometry"),
                "metadata": props
            })

        return scenes

    async def get_download_url(self, scene_id: str) -> str:
        """Get download URL for Planet scene"""
        # Implementation would activate and get download URL
        return ""

    async def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.planet.com/data/v1",
                    auth=(self.api_key, "")
                )
                return response.status_code == 200
        except Exception:
            return False


class MaxarProvider(SatelliteProvider):
    """Maxar Technologies provider"""

    def __init__(self):
        self.api_key = settings.maxar_api_key

    @property
    def name(self) -> str:
        return "maxar"

    def is_enabled(self) -> bool:
        return settings.maxar_enabled and bool(self.api_key)

    async def search(
        self,
        aoi: dict,
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: Optional[float] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search Maxar catalog"""
        if not self.is_enabled():
            return []

        # Maxar SecureWatch API would go here
        logger.warning("Maxar search not implemented")
        return []

    async def get_download_url(self, scene_id: str) -> str:
        return ""

    async def health_check(self) -> bool:
        return self.is_enabled()


class CapellaProvider(SatelliteProvider):
    """Capella Space SAR provider"""

    def __init__(self):
        self.api_key = settings.capella_api_key

    @property
    def name(self) -> str:
        return "capella"

    def is_enabled(self) -> bool:
        return settings.capella_enabled and bool(self.api_key)

    async def search(
        self,
        aoi: dict,
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: Optional[float] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search Capella SAR catalog"""
        if not self.is_enabled():
            return []

        # Capella API would go here
        logger.warning("Capella search not implemented")
        return []

    async def get_download_url(self, scene_id: str) -> str:
        return ""

    async def health_check(self) -> bool:
        return self.is_enabled()


class ProviderFactory:
    """Factory for creating and managing satellite providers"""

    def __init__(self):
        self._providers: Dict[str, SatelliteProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all available providers"""
        self._providers = {
            "sentinel": SentinelProvider(),
            "planet": PlanetProvider(),
            "maxar": MaxarProvider(),
            "capella": CapellaProvider(),
        }

    def get_provider(self, name: str) -> Optional[SatelliteProvider]:
        """Get a provider by name"""
        return self._providers.get(name)

    def get_enabled_providers(self) -> List[SatelliteProvider]:
        """Get all enabled providers"""
        return [p for p in self._providers.values() if p.is_enabled()]

    def list_providers(self) -> Dict[str, bool]:
        """List all providers and their enabled status"""
        return {name: p.is_enabled() for name, p in self._providers.items()}


# Singleton factory instance
@lru_cache
def get_provider_factory() -> ProviderFactory:
    """Get the singleton provider factory"""
    return ProviderFactory()

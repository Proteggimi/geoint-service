"""
GEOINT Service Configuration
Gestisce tutte le configurazioni del microservizio
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # Application
    app_name: str = "GEOINT Service"
    app_version: str = "1.0.0"
    debug: bool = False
    lite_mode: bool = False  # Lite mode for Render.com free tier

    # Server
    host: str = "0.0.0.0"
    port: int = 8081
    workers: int = 4

    # Database - PostGIS
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "geoint"
    postgres_user: str = "geoint"
    postgres_password: str = "geoint_secret"

    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def database_url_sync(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # MinIO / S3
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket_scenes: str = "satellite-scenes"
    minio_bucket_cog: str = "cog-tiles"
    minio_bucket_thumbnails: str = "thumbnails"

    # Satellite Providers
    # Sentinel Hub (Copernicus)
    sentinel_hub_client_id: Optional[str] = None
    sentinel_hub_client_secret: Optional[str] = None
    sentinel_hub_enabled: bool = True

    # Planet Labs
    planet_api_key: Optional[str] = None
    planet_enabled: bool = False

    # Maxar
    maxar_api_key: Optional[str] = None
    maxar_enabled: bool = False

    # Capella Space (SAR)
    capella_api_key: Optional[str] = None
    capella_enabled: bool = False

    # ML Models
    yolo_model_path: str = "models/yolov8_satellite.pt"
    detection_confidence: float = 0.25
    detection_iou_threshold: float = 0.45

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # PHP Backend Integration
    php_backend_url: str = "http://localhost:8080"
    php_auth_secret: str = "shared_secret_with_php"

    # Tile Server
    tile_cache_ttl: int = 3600  # 1 hour
    max_tile_size: int = 512


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

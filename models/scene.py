"""
Satellite Scene model
"""
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, Numeric, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geometry

from database import Base


class SatelliteScene(Base):
    """
    Represents a satellite imagery scene.
    Stores metadata and footprint for acquired scenes.
    """
    __tablename__ = "satellite_scenes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Provider info
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    scene_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # Temporal
    acquisition_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Scene properties
    cloud_cover: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    resolution_m: Mapped[Optional[float]] = mapped_column(Numeric(10, 2))
    sensor: Mapped[Optional[str]] = mapped_column(String(50))
    processing_level: Mapped[Optional[str]] = mapped_column(String(20))

    # Storage
    storage_path: Mapped[Optional[str]] = mapped_column(String(500))
    cog_url: Mapped[Optional[str]] = mapped_column(String(500))
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(500))

    # Geometry - Polygon footprint in EPSG:4326
    footprint: Mapped[str] = mapped_column(
        Geometry("POLYGON", srid=4326),
        nullable=False
    )

    # Additional metadata as JSONB
    metadata: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Band information
    bands: Mapped[Optional[list]] = mapped_column(JSONB)

    # Processing status
    is_downloaded: Mapped[bool] = mapped_column(Boolean, default=False)
    is_cog_ready: Mapped[bool] = mapped_column(Boolean, default=False)

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow
    )
    created_by: Mapped[Optional[int]] = mapped_column()  # User ID from PHP backend

    __table_args__ = (
        Index("idx_scenes_footprint", footprint, postgresql_using="gist"),
        Index("idx_scenes_acquisition", acquisition_date.desc()),
        Index("idx_scenes_provider_date", provider, acquisition_date.desc()),
    )

    def __repr__(self) -> str:
        return f"<SatelliteScene {self.provider}:{self.scene_id}>"

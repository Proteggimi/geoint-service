"""
Object Detection model
"""
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, Numeric, Boolean, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from geoalchemy2 import Geometry

from database import Base


class ObjectDetection(Base):
    """
    Represents an object detected in satellite imagery.
    Includes location, classification, and confidence.
    """
    __tablename__ = "object_detections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Link to scene
    scene_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("satellite_scenes.id", ondelete="CASCADE"),
        nullable=False
    )

    # Detection type
    detection_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    class_name: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)

    # Bounding box in pixel coordinates
    bbox_pixel: Mapped[Optional[dict]] = mapped_column(JSONB)
    # Format: {"x": 100, "y": 200, "width": 50, "height": 30}

    # Georeferenced location - Point centroid
    location: Mapped[str] = mapped_column(
        Geometry("POINT", srid=4326),
        nullable=False
    )

    # Georeferenced footprint - Polygon bounding box
    footprint: Mapped[Optional[str]] = mapped_column(
        Geometry("POLYGON", srid=4326)
    )

    # Additional attributes
    attributes: Mapped[Optional[dict]] = mapped_column(JSONB)
    # Can include: color, orientation, size_m2, etc.

    # Linking to OSINT cases
    case_id: Mapped[Optional[int]] = mapped_column()  # From PHP cases table

    # Timestamps
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow
    )

    # Verification
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verified_by: Mapped[Optional[int]] = mapped_column()
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_detections_location", location, postgresql_using="gist"),
        Index("idx_detections_scene", scene_id),
        Index("idx_detections_type_conf", detection_type, confidence.desc()),
    )

    def __repr__(self) -> str:
        return f"<ObjectDetection {self.class_name} ({self.confidence:.2f})>"

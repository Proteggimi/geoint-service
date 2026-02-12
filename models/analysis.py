"""
Change Analysis model
"""
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geometry

from database import Base


class ChangeAnalysis(Base):
    """
    Represents a change detection analysis between two scenes.
    Stores the AOI, method, and results.
    """
    __tablename__ = "change_analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Scenes being compared
    before_scene_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("satellite_scenes.id", ondelete="SET NULL"),
        nullable=True
    )
    after_scene_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("satellite_scenes.id", ondelete="SET NULL"),
        nullable=True
    )

    # Area of interest for analysis
    aoi: Mapped[str] = mapped_column(
        Geometry("POLYGON", srid=4326),
        nullable=False
    )

    # Analysis configuration
    method: Mapped[str] = mapped_column(String(50), nullable=False)
    # Methods: 'ratio', 'difference', 'cvaps', 'coherence'

    parameters: Mapped[Optional[dict]] = mapped_column(JSONB)
    # Parameters: threshold, window_size, etc.

    # Results
    change_mask_url: Mapped[Optional[str]] = mapped_column(String(500))
    statistics: Mapped[Optional[dict]] = mapped_column(JSONB)
    # Stats: total_change_area_km2, increase_pct, decrease_pct, etc.

    # Status
    status: Mapped[str] = mapped_column(String(20), default="pending")
    # Status: pending, processing, completed, failed
    error_message: Mapped[Optional[str]] = mapped_column(String(500))

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[Optional[int]] = mapped_column()

    # Link to case
    case_id: Mapped[Optional[int]] = mapped_column()

    __table_args__ = (
        Index("idx_changes_aoi", aoi, postgresql_using="gist"),
        Index("idx_changes_status", status),
    )

    def __repr__(self) -> str:
        return f"<ChangeAnalysis {self.id} ({self.method})>"

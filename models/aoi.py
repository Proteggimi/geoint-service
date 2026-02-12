"""
Area of Interest model
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Text, Integer, Boolean, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geometry

from database import Base


class AreaOfInterest(Base):
    """
    Represents a persistent Area of Interest for monitoring.
    Users can save AOIs and set up automated monitoring.
    """
    __tablename__ = "areas_of_interest"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Ownership
    user_id: Mapped[int] = mapped_column(nullable=False, index=True)
    team_id: Mapped[Optional[int]] = mapped_column(index=True)

    # AOI details
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Geometry - can be Point, Polygon, or MultiPolygon
    geometry: Mapped[str] = mapped_column(
        Geometry("GEOMETRY", srid=4326),
        nullable=False
    )

    # Monitoring settings
    monitor_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    monitor_interval_hours: Mapped[int] = mapped_column(Integer, default=168)  # Weekly
    last_checked: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    next_check: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Alert settings
    alert_on_change: Mapped[bool] = mapped_column(Boolean, default=True)
    alert_threshold: Mapped[float] = mapped_column(default=0.1)  # 10% change
    alert_email: Mapped[Optional[str]] = mapped_column(String(255))

    # Provider preferences
    preferred_providers: Mapped[Optional[list]] = mapped_column(JSONB)
    # ["sentinel", "planet"]

    max_cloud_cover: Mapped[int] = mapped_column(Integer, default=30)

    # Styling for map display
    style: Mapped[Optional[dict]] = mapped_column(JSONB)
    # {"color": "#ff0000", "fillOpacity": 0.3}

    # Link to case
    case_id: Mapped[Optional[int]] = mapped_column()

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_aoi_geometry", geometry, postgresql_using="gist"),
        Index("idx_aoi_user", user_id),
        Index("idx_aoi_monitoring", monitor_enabled, next_check),
    )

    def __repr__(self) -> str:
        return f"<AreaOfInterest {self.name}>"

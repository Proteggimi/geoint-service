"""
Database connection and session management for PostGIS
Supports lite mode with SQLite for Render.com free tier
"""
import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
import structlog

from config import get_settings

logger = structlog.get_logger()
settings = get_settings()


def get_database_url() -> str:
    """Get database URL based on mode and environment"""
    # Check for Render.com DATABASE_URL
    render_db_url = os.environ.get("DATABASE_URL")
    if render_db_url:
        # Render provides postgres:// but asyncpg needs postgresql+asyncpg://
        if render_db_url.startswith("postgres://"):
            render_db_url = render_db_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif render_db_url.startswith("postgresql://"):
            render_db_url = render_db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return render_db_url

    # Lite mode uses SQLite
    if settings.lite_mode:
        return "sqlite+aiosqlite:///./geoint_lite.db"

    # Default PostGIS connection
    return settings.database_url


# Create async engine
database_url = get_database_url()
is_sqlite = database_url.startswith("sqlite")

engine_kwargs = {
    "echo": settings.debug,
}

if not is_sqlite:
    engine_kwargs.update({
        "pool_size": 5,
        "max_overflow": 10,
        "pool_pre_ping": True,
    })

engine = create_async_engine(database_url, **engine_kwargs)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all models"""
    pass


async def init_db():
    """Initialize database connection and create tables"""
    # In lite mode, skip database initialization (using mock data only)
    if settings.lite_mode:
        logger.info("Lite mode enabled - skipping database initialization, using mock data")
        return

    async with engine.begin() as conn:
        # Enable PostGIS if using PostgreSQL
        if not is_sqlite:
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_raster"))
                logger.info("PostGIS extensions enabled")
            except Exception as e:
                logger.warning("PostGIS not available, using basic mode", error=str(e))

        # Import models to register them with Base
        # Only import in full mode to avoid GeoAlchemy2 dependency
        from models import scene, detection, analysis, aoi

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

    db_type = "SQLite" if is_sqlite else "PostgreSQL"
    logger.info(f"Database initialized ({db_type})")


async def close_db():
    """Close database connections"""
    await engine.dispose()
    logger.info("Database connections closed")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_connection() -> bool:
    """Check if database is accessible"""
    # In lite mode, always return True (no real database)
    if settings.lite_mode:
        return True

    try:
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error("Database connection failed", error=str(e))
        return False

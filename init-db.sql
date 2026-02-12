-- PostGIS initialization script
-- Creates extensions and initial schema for GEOINT database

-- Enable PostGIS extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE geoint TO geoint;

-- Create spatial reference systems if not exists
-- (PostGIS includes these by default, but just in case)

-- Add any custom initialization here
-- The actual table creation is handled by SQLAlchemy models

-- Create index on spatial_ref_sys for faster lookups
CREATE INDEX IF NOT EXISTS idx_spatial_ref_sys_srid ON spatial_ref_sys(srid);

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'PostGIS database initialized successfully';
END $$;

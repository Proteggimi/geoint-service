"""
Detection Service - Object detection in satellite imagery using YOLO
"""
import io
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
import numpy as np
from PIL import Image
import structlog

from config import get_settings

logger = structlog.get_logger()
settings = get_settings()

# Detection class mapping
DETECTION_CLASSES = {
    # Vehicles
    0: {"type": "vehicle", "name": "car", "color": (59, 130, 246)},
    1: {"type": "vehicle", "name": "truck", "color": (59, 130, 246)},
    2: {"type": "vehicle", "name": "bus", "color": (59, 130, 246)},
    3: {"type": "vehicle", "name": "tank", "color": (239, 68, 68)},
    4: {"type": "vehicle", "name": "apc", "color": (239, 68, 68)},

    # Ships
    10: {"type": "ship", "name": "cargo", "color": (16, 185, 129)},
    11: {"type": "ship", "name": "tanker", "color": (16, 185, 129)},
    12: {"type": "ship", "name": "container", "color": (16, 185, 129)},
    13: {"type": "ship", "name": "military", "color": (239, 68, 68)},
    14: {"type": "ship", "name": "fishing", "color": (16, 185, 129)},

    # Aircraft
    20: {"type": "aircraft", "name": "commercial", "color": (168, 85, 247)},
    21: {"type": "aircraft", "name": "military_jet", "color": (239, 68, 68)},
    22: {"type": "aircraft", "name": "helicopter", "color": (168, 85, 247)},
    23: {"type": "aircraft", "name": "drone", "color": (168, 85, 247)},

    # Buildings
    30: {"type": "building", "name": "residential", "color": (251, 146, 60)},
    31: {"type": "building", "name": "commercial", "color": (251, 146, 60)},
    32: {"type": "building", "name": "industrial", "color": (251, 146, 60)},
    33: {"type": "building", "name": "military", "color": (239, 68, 68)},

    # Infrastructure
    40: {"type": "infrastructure", "name": "road", "color": (156, 163, 175)},
    41: {"type": "infrastructure", "name": "bridge", "color": (156, 163, 175)},
    42: {"type": "infrastructure", "name": "runway", "color": (156, 163, 175)},
    43: {"type": "infrastructure", "name": "port", "color": (156, 163, 175)},
    44: {"type": "infrastructure", "name": "solar_farm", "color": (234, 179, 8)},
}


class ObjectDetector:
    """YOLO-based object detector for satellite imagery"""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path or settings.yolo_model_path
        self._load_model()

    def _load_model(self):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO

            # Try to load custom model first
            try:
                self.model = YOLO(self.model_path)
                logger.info("Loaded custom YOLO model", path=self.model_path)
            except Exception:
                # Fall back to pretrained model
                self.model = YOLO("yolov8n.pt")
                logger.info("Loaded pretrained YOLOv8n model")

        except ImportError:
            logger.warning("ultralytics not available, detection will use mock data")
            self.model = None

    async def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        detection_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run object detection on an image.

        Args:
            image: Input image as numpy array (H, W, C) in RGB
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            detection_types: Filter to specific types (e.g., ["vehicle", "ship"])

        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            return self._generate_mock_detections(image.shape, detection_types)

        # Run inference
        results = self.model(
            image,
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                # Get class info
                class_info = DETECTION_CLASSES.get(class_id, {
                    "type": "unknown",
                    "name": f"class_{class_id}",
                    "color": (128, 128, 128)
                })

                # Filter by detection type
                if detection_types and class_info["type"] not in detection_types:
                    continue

                detections.append({
                    "class_id": class_id,
                    "detection_type": class_info["type"],
                    "class_name": class_info["name"],
                    "confidence": confidence,
                    "bbox_pixel": {
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                        "width": bbox[2] - bbox[0],
                        "height": bbox[3] - bbox[1],
                        "center_x": (bbox[0] + bbox[2]) / 2,
                        "center_y": (bbox[1] + bbox[3]) / 2,
                    },
                    "color": class_info["color"],
                })

        return detections

    def _generate_mock_detections(
        self,
        image_shape: Tuple,
        detection_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate mock detections for testing"""
        import random

        height, width = image_shape[:2]
        num_detections = random.randint(3, 15)

        mock_classes = [
            {"type": "vehicle", "name": "car"},
            {"type": "vehicle", "name": "truck"},
            {"type": "ship", "name": "cargo"},
            {"type": "building", "name": "industrial"},
        ]

        if detection_types:
            mock_classes = [c for c in mock_classes if c["type"] in detection_types]

        detections = []
        for _ in range(num_detections):
            if not mock_classes:
                break

            cls = random.choice(mock_classes)
            x1 = random.randint(50, width - 100)
            y1 = random.randint(50, height - 100)
            w = random.randint(20, 80)
            h = random.randint(20, 80)

            detections.append({
                "class_id": 0,
                "detection_type": cls["type"],
                "class_name": cls["name"],
                "confidence": random.uniform(0.5, 0.95),
                "bbox_pixel": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x1 + w,
                    "y2": y1 + h,
                    "width": w,
                    "height": h,
                    "center_x": x1 + w / 2,
                    "center_y": y1 + h / 2,
                },
                "color": (59, 130, 246),
            })

        return detections


# Global detector instance
_detector: Optional[ObjectDetector] = None


def get_detector() -> ObjectDetector:
    """Get or create the global detector instance"""
    global _detector
    if _detector is None:
        _detector = ObjectDetector()
    return _detector


async def enqueue_detection(
    task_id: str,
    scene_id: UUID,
    cog_url: str,
    aoi: Optional[Dict] = None,
    detection_types: Optional[List[str]] = None,
    confidence_threshold: float = 0.25
):
    """
    Run detection on a satellite scene.
    Uses sliding window approach for large images.
    """
    from database import async_session_maker
    from models.detection import ObjectDetection
    from models.scene import SatelliteScene
    from sqlalchemy import select
    from geoalchemy2.functions import ST_MakePoint, ST_SetSRID

    logger.info("Starting detection job",
                task_id=task_id,
                scene_id=str(scene_id))

    detector = get_detector()

    try:
        # Load the scene image
        image, transform = await _load_scene_image(cog_url, aoi)

        # Run detection with sliding window
        tile_size = 640
        overlap = 64
        all_detections = []

        height, width = image.shape[:2]

        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                # Extract tile
                tile = image[
                    y:min(y + tile_size, height),
                    x:min(x + tile_size, width)
                ]

                # Skip if tile is too small
                if tile.shape[0] < 100 or tile.shape[1] < 100:
                    continue

                # Run detection
                tile_detections = await detector.detect(
                    tile,
                    confidence_threshold=confidence_threshold,
                    detection_types=detection_types
                )

                # Adjust coordinates to global image space
                for det in tile_detections:
                    det["bbox_pixel"]["x1"] += x
                    det["bbox_pixel"]["x2"] += x
                    det["bbox_pixel"]["y1"] += y
                    det["bbox_pixel"]["y2"] += y
                    det["bbox_pixel"]["center_x"] += x
                    det["bbox_pixel"]["center_y"] += y

                    # Convert pixel to geo coordinates
                    if transform:
                        geo_x, geo_y = _pixel_to_geo(
                            det["bbox_pixel"]["center_x"],
                            det["bbox_pixel"]["center_y"],
                            transform
                        )
                        det["geo_location"] = {"lng": geo_x, "lat": geo_y}

                all_detections.extend(tile_detections)

        # Apply NMS across tiles
        final_detections = _global_nms(all_detections)

        # Save to database
        async with async_session_maker() as session:
            for det in final_detections:
                geo_loc = det.get("geo_location", {"lng": 0, "lat": 0})

                detection = ObjectDetection(
                    scene_id=scene_id,
                    detection_type=det["detection_type"],
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    bbox_pixel=det["bbox_pixel"],
                    location=f"SRID=4326;POINT({geo_loc['lng']} {geo_loc['lat']})",
                    attributes={"color": det["color"]},
                )
                session.add(detection)

            await session.commit()

        logger.info("Detection completed",
                    task_id=task_id,
                    detections=len(final_detections))

    except Exception as e:
        logger.error("Detection failed",
                     task_id=task_id,
                     error=str(e))
        raise


async def _load_scene_image(
    cog_url: str,
    aoi: Optional[Dict] = None
) -> Tuple[np.ndarray, Optional[Any]]:
    """Load scene image from COG"""
    try:
        import rasterio
        from rasterio.io import MemoryFile
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(cog_url)
            response.raise_for_status()
            data = response.content

        with MemoryFile(data) as memfile:
            with memfile.open() as src:
                # Read RGB bands
                if src.count >= 3:
                    rgb = np.dstack([src.read(4), src.read(3), src.read(2)])
                else:
                    rgb = np.dstack([src.read(1)] * 3)

                # Normalize to 0-255
                rgb = (rgb / rgb.max() * 255).astype(np.uint8)

                return rgb, src.transform

    except Exception as e:
        logger.error("Failed to load scene image", error=str(e))
        # Return placeholder image
        return np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8), None


def _pixel_to_geo(x: float, y: float, transform) -> Tuple[float, float]:
    """Convert pixel coordinates to geographic coordinates"""
    try:
        from rasterio.transform import xy
        geo_x, geo_y = xy(transform, y, x)
        return float(geo_x), float(geo_y)
    except Exception:
        return 0.0, 0.0


def _global_nms(
    detections: List[Dict],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """Apply Non-Maximum Suppression across all detections"""
    if not detections:
        return []

    # Sort by confidence
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)

        detections = [
            d for d in detections
            if _calculate_iou(best["bbox_pixel"], d["bbox_pixel"]) < iou_threshold
        ]

    return keep


def _calculate_iou(box1: Dict, box2: Dict) -> float:
    """Calculate Intersection over Union for two boxes"""
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

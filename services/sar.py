"""
SAR Processing Service - Synthetic Aperture Radar image processing
Includes speckle filtering, calibration, and terrain correction
"""
import asyncio
import numpy as np
from typing import Optional, Tuple, Literal
from enum import Enum
import structlog

logger = structlog.get_logger()


# ===========================================
# Enums
# ===========================================

class Polarization(str, Enum):
    VV = "VV"
    VH = "VH"
    VV_VH = "VV+VH"
    VV_DIV_VH = "VV/VH"


class SpeckleFilter(str, Enum):
    NONE = "none"
    LEE = "lee"
    FROST = "frost"
    GAMMA_MAP = "gamma_map"
    REFINED_LEE = "refined_lee"


class Calibration(str, Enum):
    SIGMA0 = "sigma0"
    GAMMA0 = "gamma0"
    BETA0 = "beta0"


# ===========================================
# Speckle Filters
# ===========================================

def lee_filter(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Lee speckle filter.

    The Lee filter is an adaptive filter that estimates the local mean
    and variance to reduce speckle while preserving edges.
    """
    from scipy.ndimage import uniform_filter, generic_filter

    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    # Calculate local mean and variance
    img_mean = uniform_filter(image.astype(np.float64), size=window_size)
    img_sqr_mean = uniform_filter(image.astype(np.float64)**2, size=window_size)
    img_variance = img_sqr_mean - img_mean**2

    # Estimate noise variance (assume multiplicative noise model)
    overall_variance = np.var(image)

    # Calculate weight
    weight = img_variance / (img_variance + overall_variance + 1e-10)

    # Apply filter
    result = img_mean + weight * (image - img_mean)

    return result.astype(image.dtype)


def frost_filter(image: np.ndarray, window_size: int = 5, damping: float = 2.0) -> np.ndarray:
    """
    Frost speckle filter.

    The Frost filter uses an exponentially decaying kernel based on
    local coefficient of variation.
    """
    from scipy.ndimage import generic_filter

    if window_size % 2 == 0:
        window_size += 1

    half_size = window_size // 2

    def frost_kernel(window):
        center = len(window) // 2
        window_2d = window.reshape(window_size, window_size)

        mean_val = np.mean(window_2d)
        var_val = np.var(window_2d)

        if mean_val == 0:
            return window_2d[half_size, half_size]

        cv = np.sqrt(var_val) / mean_val  # Coefficient of variation

        # Create distance weights
        y, x = np.ogrid[-half_size:half_size+1, -half_size:half_size+1]
        dist = np.sqrt(x**2 + y**2)

        # Exponential kernel
        weights = np.exp(-damping * cv * dist)
        weights = weights / np.sum(weights)

        return np.sum(window_2d * weights)

    result = generic_filter(
        image.astype(np.float64),
        frost_kernel,
        size=window_size
    )

    return result.astype(image.dtype)


def gamma_map_filter(image: np.ndarray, window_size: int = 5, nlooks: int = 1) -> np.ndarray:
    """
    Gamma MAP (Maximum A Posteriori) speckle filter.

    Based on the assumption that the original image follows a Gamma distribution.
    """
    from scipy.ndimage import uniform_filter

    if window_size % 2 == 0:
        window_size += 1

    # Calculate local statistics
    img_mean = uniform_filter(image.astype(np.float64), size=window_size)
    img_sqr_mean = uniform_filter(image.astype(np.float64)**2, size=window_size)
    img_variance = img_sqr_mean - img_mean**2

    # Coefficient of variation
    cu = 1.0 / np.sqrt(nlooks)  # Noise coefficient of variation
    ci = np.sqrt(img_variance) / (img_mean + 1e-10)

    # Maximum coefficient of variation
    cmax = np.sqrt(2) * cu

    # Calculate alpha
    alpha = (1 + cu**2) / (ci**2 - cu**2 + 1e-10)

    # Apply filter based on ci value
    result = np.where(
        ci <= cu,
        img_mean,
        np.where(
            ci >= cmax,
            image,
            img_mean * alpha + image * (1 - alpha)
        )
    )

    return result.astype(image.dtype)


def refined_lee_filter(image: np.ndarray, window_size: int = 7, nlooks: int = 1) -> np.ndarray:
    """
    Refined Lee filter with edge-preserving capabilities.

    Uses directional windows to better preserve edges and linear features.
    """
    from scipy.ndimage import uniform_filter

    if window_size % 2 == 0:
        window_size += 1

    half_size = window_size // 2

    # Define 8 directional templates
    templates = []
    for angle in range(0, 180, 45):
        template = np.zeros((window_size, window_size))
        rad = np.radians(angle)
        for i in range(window_size):
            for j in range(window_size):
                di = i - half_size
                dj = j - half_size
                # Distance from line through center at angle
                dist = abs(di * np.cos(rad) - dj * np.sin(rad))
                if dist <= 1.5:
                    template[i, j] = 1
        templates.append(template / np.sum(template))

    # Apply each template and select best result based on variance
    results = []
    for template in templates:
        from scipy.ndimage import convolve
        local_mean = convolve(image.astype(np.float64), template)
        results.append(local_mean)

    # Stack results and compute variance for each
    results_stack = np.stack(results, axis=0)

    # Use result with minimum variance (most homogeneous)
    variances = np.var(results_stack, axis=(1, 2), keepdims=True)
    min_var_idx = np.argmin(variances.flatten())

    # Apply Lee filter with selected direction
    img_mean = results[min_var_idx]
    img_sqr_mean = uniform_filter(image.astype(np.float64)**2, size=window_size)
    img_variance = img_sqr_mean - img_mean**2

    cu = 1.0 / np.sqrt(nlooks)
    overall_variance = cu**2 * img_mean**2

    weight = img_variance / (img_variance + overall_variance + 1e-10)
    weight = np.clip(weight, 0, 1)

    result = img_mean + weight * (image - img_mean)

    return result.astype(image.dtype)


# ===========================================
# Calibration Functions
# ===========================================

def apply_calibration(
    image: np.ndarray,
    calibration_type: Calibration,
    metadata: dict
) -> np.ndarray:
    """
    Apply radiometric calibration to SAR data.

    Converts digital numbers to backscatter coefficient.
    """
    # Extract calibration constants from metadata
    # These would come from the SAR product metadata
    k = metadata.get('calibration_constant', 1.0)
    theta_i = np.radians(metadata.get('incidence_angle', 30.0))

    # Convert to linear scale if in dB
    if metadata.get('is_db', False):
        image_linear = 10 ** (image / 10)
    else:
        image_linear = image.astype(np.float64)

    if calibration_type == Calibration.SIGMA0:
        # Sigma nought (σ°) - radar cross section per unit area
        result = image_linear * k

    elif calibration_type == Calibration.GAMMA0:
        # Gamma nought (γ°) - normalized to local incidence angle
        result = image_linear * k / np.cos(theta_i)

    elif calibration_type == Calibration.BETA0:
        # Beta nought (β°) - normalized to slant range
        result = image_linear * k * np.sin(theta_i)
    else:
        result = image_linear

    return result


def to_db(image: np.ndarray, min_val: float = 1e-10) -> np.ndarray:
    """Convert linear values to decibels."""
    return 10 * np.log10(np.maximum(image, min_val))


def from_db(image: np.ndarray) -> np.ndarray:
    """Convert decibels to linear values."""
    return 10 ** (image / 10)


# ===========================================
# SAR Processing Service
# ===========================================

class SARProcessor:
    """
    Service for processing Synthetic Aperture Radar imagery.
    """

    def __init__(self):
        self.logger = structlog.get_logger()

    async def apply_speckle_filter(
        self,
        image: np.ndarray,
        filter_type: SpeckleFilter,
        window_size: int = 5
    ) -> np.ndarray:
        """
        Apply speckle filtering to SAR image.
        """
        self.logger.info(
            "Applying speckle filter",
            filter=filter_type.value,
            window_size=window_size
        )

        # Run filter in executor to not block event loop
        loop = asyncio.get_event_loop()

        if filter_type == SpeckleFilter.NONE:
            return image
        elif filter_type == SpeckleFilter.LEE:
            result = await loop.run_in_executor(
                None, lee_filter, image, window_size
            )
        elif filter_type == SpeckleFilter.FROST:
            result = await loop.run_in_executor(
                None, frost_filter, image, window_size
            )
        elif filter_type == SpeckleFilter.GAMMA_MAP:
            result = await loop.run_in_executor(
                None, gamma_map_filter, image, window_size
            )
        elif filter_type == SpeckleFilter.REFINED_LEE:
            result = await loop.run_in_executor(
                None, refined_lee_filter, image, window_size
            )
        else:
            result = image

        return result

    async def calibrate(
        self,
        image: np.ndarray,
        calibration_type: Calibration,
        metadata: dict
    ) -> np.ndarray:
        """
        Apply radiometric calibration.
        """
        self.logger.info(
            "Applying calibration",
            type=calibration_type.value
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, apply_calibration, image, calibration_type, metadata
        )

        return result

    async def terrain_correction(
        self,
        image: np.ndarray,
        dem: Optional[np.ndarray] = None,
        metadata: dict = {}
    ) -> np.ndarray:
        """
        Apply terrain correction using DEM.

        Note: Full terrain correction requires orbit data and DEM.
        This is a simplified version.
        """
        self.logger.info("Applying terrain correction")

        if dem is None:
            # Without DEM, just return the original image
            self.logger.warning("No DEM provided, skipping terrain correction")
            return image

        # Simplified terrain correction
        # In practice, this would use Range-Doppler terrain correction
        # with proper orbit and DEM data

        # Calculate local incidence angle from DEM slope
        from scipy.ndimage import sobel

        dx = sobel(dem, axis=1)
        dy = sobel(dem, axis=0)
        slope = np.sqrt(dx**2 + dy**2)

        # Adjust backscatter based on local slope
        # This is a simplification - real TC is more complex
        correction_factor = np.cos(slope) / np.cos(np.radians(metadata.get('incidence_angle', 30)))

        return image * correction_factor

    async def compute_polarimetric_decomposition(
        self,
        vv: np.ndarray,
        vh: np.ndarray
    ) -> dict:
        """
        Compute simple polarimetric decomposition.

        Returns VV/VH ratio and other derived products.
        """
        self.logger.info("Computing polarimetric decomposition")

        # VV/VH ratio (useful for vegetation/moisture analysis)
        ratio = vv / (vh + 1e-10)

        # Cross-pol ratio normalized
        norm_diff = (vv - vh) / (vv + vh + 1e-10)

        # Simple Radar Vegetation Index (RVI)
        # RVI = 4 * VH / (VV + VH)
        rvi = 4 * vh / (vv + vh + 1e-10)

        return {
            'vv_vh_ratio': ratio,
            'normalized_difference': norm_diff,
            'radar_vegetation_index': rvi
        }

    async def compute_coherence(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        window_size: int = 5
    ) -> np.ndarray:
        """
        Compute interferometric coherence between two SAR images.

        Coherence is useful for:
        - Change detection
        - Surface stability analysis
        - InSAR quality assessment
        """
        self.logger.info("Computing coherence", window_size=window_size)

        from scipy.ndimage import uniform_filter

        # Ensure complex data
        if not np.iscomplexobj(image1):
            image1 = image1.astype(np.complex128)
        if not np.iscomplexobj(image2):
            image2 = image2.astype(np.complex128)

        # Compute coherence
        # γ = |<s1 * s2*>| / sqrt(<|s1|²> * <|s2|²>)

        cross = image1 * np.conj(image2)
        power1 = np.abs(image1) ** 2
        power2 = np.abs(image2) ** 2

        # Apply smoothing
        cross_smooth = uniform_filter(cross.real, size=window_size) + \
                       1j * uniform_filter(cross.imag, size=window_size)
        power1_smooth = uniform_filter(power1, size=window_size)
        power2_smooth = uniform_filter(power2, size=window_size)

        coherence = np.abs(cross_smooth) / \
                    np.sqrt(power1_smooth * power2_smooth + 1e-10)

        return np.clip(coherence, 0, 1)

    async def process_scene(
        self,
        image: np.ndarray,
        polarization: Polarization,
        speckle_filter: SpeckleFilter,
        filter_window_size: int,
        calibration: Calibration,
        terrain_correction: bool,
        metadata: dict,
        dem: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Full SAR processing pipeline.
        """
        self.logger.info(
            "Processing SAR scene",
            polarization=polarization.value,
            filter=speckle_filter.value,
            calibration=calibration.value
        )

        result = image.copy()
        processing_log = []

        # 1. Calibration
        result = await self.calibrate(result, calibration, metadata)
        processing_log.append(f"Applied {calibration.value} calibration")

        # 2. Terrain correction
        if terrain_correction and dem is not None:
            result = await self.terrain_correction(result, dem, metadata)
            processing_log.append("Applied terrain correction")

        # 3. Speckle filtering
        if speckle_filter != SpeckleFilter.NONE:
            result = await self.apply_speckle_filter(
                result, speckle_filter, filter_window_size
            )
            processing_log.append(f"Applied {speckle_filter.value} filter")

        # 4. Convert to dB for visualization
        result_db = to_db(result)

        stats = {
            'min_db': float(np.nanmin(result_db)),
            'max_db': float(np.nanmax(result_db)),
            'mean_db': float(np.nanmean(result_db)),
            'std_db': float(np.nanstd(result_db)),
            'processing_log': processing_log
        }

        return result_db, stats


# ===========================================
# Singleton Instance
# ===========================================

sar_processor = SARProcessor()

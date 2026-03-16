import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_experiment1(
    raw_data: np.ndarray,
    labels: np.ndarray,
    cloud_mask: np.ndarray,
    cloud_threshold: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    stats = {"raw_parcels": len(labels)}

    valid_mask = ~_detect_duplicates(raw_data)
    raw_data, labels = raw_data[valid_mask], labels[valid_mask]
    stats["after_deduplication"] = len(labels)

    raw_data = _impute_cloud_contaminated(raw_data, cloud_mask, cloud_threshold)
    stats["after_cloud_screening"] = len(labels)

    raw_data, labels = _interpolate_and_qa(raw_data, labels)
    stats["after_interpolation_qa"] = len(labels)

    raw_data = _normalise_spectral(raw_data)
    stats["final_labelled"] = len(labels)

    logger.info(f"Experiment 1 preprocessing complete: {stats}")
    return raw_data, labels, stats


def preprocess_experiment2(
    sentinel2: np.ndarray,
    lidar_dtm: np.ndarray,
    land_cover: np.ndarray,
    soil_drainage: np.ndarray,
    flood_labels: np.ndarray,
    cloud_mask: np.ndarray,
    cloud_threshold: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    stats = {"raw_pixels": sentinel2.shape[0] if sentinel2.ndim > 1 else 0}

    sentinel2 = _apply_cloud_mask(sentinel2, cloud_mask, cloud_threshold)
    stats["after_cloud_masking"] = sentinel2.shape[0] if sentinel2.ndim > 1 else 0

    fused, flood_labels = _fuse_sources(sentinel2, lidar_dtm, land_cover, soil_drainage, flood_labels)
    stats["after_coregistration_fusion"] = fused.shape[0] if fused.ndim > 1 else 0

    fused, flood_labels = _label_qa(fused, flood_labels)
    stats["after_label_qa"] = fused.shape[0] if fused.ndim > 1 else 0

    fused = _normalise_spectral(fused)
    stats["final_labelled_pixels"] = fused.shape[0] if fused.ndim > 1 else 0

    logger.info(f"Experiment 2 preprocessing complete: {stats}")
    return fused, flood_labels, stats


def _detect_duplicates(data: np.ndarray) -> np.ndarray:
    if data.ndim < 2:
        return np.zeros(len(data), dtype=bool)
    _, unique_idx = np.unique(data.reshape(len(data), -1), axis=0, return_index=True)
    mask = np.ones(len(data), dtype=bool)
    mask[unique_idx] = False
    return mask


def _impute_cloud_contaminated(
    data: np.ndarray, cloud_mask: np.ndarray, threshold: float
) -> np.ndarray:
    if cloud_mask is None or data.ndim < 3:
        return data
    contaminated = cloud_mask > threshold
    for t in range(1, data.shape[1] - 1):
        bad = contaminated[:, t]
        if bad.any():
            data[bad, t] = (data[bad, t - 1] + data[bad, t + 1]) / 2.0
    return data


def _apply_cloud_mask(data: np.ndarray, cloud_mask: np.ndarray, threshold: float) -> np.ndarray:
    if cloud_mask is None:
        return data
    valid = cloud_mask <= threshold
    return data[valid] if data.ndim > 1 else data


def _interpolate_and_qa(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = labels >= 0
    return data[valid], labels[valid]


def _fuse_sources(
    sentinel2: np.ndarray,
    lidar_dtm: np.ndarray,
    land_cover: np.ndarray,
    soil_drainage: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(sentinel2) if sentinel2.ndim > 1 else 0
    if n == 0:
        return sentinel2, labels

    ndvi = _safe_ratio(sentinel2[:, 7], sentinel2[:, 3])
    ndwi = _safe_ratio(sentinel2[:, 2], sentinel2[:, 7])
    ndbi = _safe_ratio(sentinel2[:, 10], sentinel2[:, 7])

    s2_full = np.concatenate([sentinel2, ndvi[:, None], ndwi[:, None], ndbi[:, None]], axis=1)

    dtm_ch = lidar_dtm[:, None] if lidar_dtm.ndim == 1 else lidar_dtm
    perm = 0.6 * land_cover + 0.4 * soil_drainage
    perm_ch = perm[:, None] if perm.ndim == 1 else perm

    fused = np.concatenate([s2_full, dtm_ch, perm_ch], axis=1)
    return fused, labels


def _label_qa(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = (labels >= 0) & (labels < 5)
    return data[valid], labels[valid]


def _normalise_spectral(data: np.ndarray) -> np.ndarray:
    if data.ndim < 2:
        return data
    min_val = data.min(axis=0, keepdims=True)
    max_val = data.max(axis=0, keepdims=True)
    denom = np.where(max_val - min_val == 0, 1.0, max_val - min_val)
    return (data - min_val) / denom


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    return np.where(denom == 0, 0.0, (a - b) / denom)

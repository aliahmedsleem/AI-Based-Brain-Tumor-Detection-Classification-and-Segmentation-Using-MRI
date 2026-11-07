"""Research-only preprocessing and model readiness. No tumor inference is performed."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PreprocessSummary:
    shape: tuple[int, int, int]
    nonzero_voxels: int
    normalized_mean: float
    normalized_std: float
    clipping_percentiles: tuple[float, float]


def preprocess_mri(volume: np.ndarray) -> tuple[np.ndarray, PreprocessSummary]:
    """Finite-only, nonzero-mask percentile clipping and z-score normalization.

    Returns float32 normalized volume and descriptive statistics.
    This is a generic baseline, NOT BraTS modality-specific preprocessing.
    """
    if volume.ndim != 3 or volume.size == 0:
        raise ValueError("Expected nonempty 3D volume")
    if volume.size > 67_108_864:
        raise ValueError("Volume exceeds preprocessing memory budget")
    array = np.asarray(volume, dtype=np.float32)
    valid = np.isfinite(array) & (array != 0)
    if not np.any(valid):
        raise ValueError("Volume has no finite nonzero voxels")
    values = array[valid]
    low, high = (float(v) for v in np.percentile(values, [0.5, 99.5]))
    if high <= low:
        high = low + 1.0
    clipped = np.clip(values, low, high)
    mean = float(np.mean(clipped, dtype=np.float64))
    std = float(np.std(clipped, dtype=np.float64))
    if std < 1e-8:
        raise ValueError("MRI intensities have insufficient variation")
    output = np.zeros(array.shape, dtype=np.float32)
    output[valid] = (clipped - mean) / std
    summary = PreprocessSummary(
        shape=tuple(int(x) for x in array.shape),
        nonzero_voxels=int(np.count_nonzero(valid)),
        normalized_mean=float(np.mean(output[valid], dtype=np.float64)),
        normalized_std=float(np.std(output[valid], dtype=np.float64)),
        clipping_percentiles=(0.5, 99.5),
    )
    return output, summary


def model_readiness() -> dict:
    """Explicitly disclose missing weights rather than producing invented predictions."""
    return {
        "pipeline": "research-preprocessing-v1",
        "preprocessing_available": True,
        "model_loaded": False,
        "classification_available": False,
        "segmentation_available": False,
        "detection_available": False,
        "required_before_inference": [
            "Licensed, de-identified MRI dataset and verified modalities",
            "Patient-level train/validation/test split",
            "Documented preprocessing and label schema",
            "Trained model weights with independent evaluation",
        ],
    }

"""Geometry-aware research mask measurements.

Bounding-box dimensions describe the mask extent, not true tumor diameters.
Anatomical lobes and hemispheric laterality require validated registration.
"""
import numpy as np


def describe_mask(mask, affine):
    mask = np.asarray(mask, dtype=bool)
    affine = np.asarray(affine, dtype=float)
    if mask.ndim != 3 or affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Invalid mask geometry")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise ValueError("Invalid voxel spacing")
    count = int(np.count_nonzero(mask))
    result = {
        "segmented_voxels": count,
        "estimated_mask_volume_ml": round(count * abs(float(np.linalg.det(affine[:3, :3]))) / 1000, 3),
        "voxel_spacing_mm": [round(float(x), 4) for x in spacing],
        "anatomical_region": None,
        "hemisphere": "Undetermined — atlas registration and expert review required",
        "location_method": "Image-space bounding box; not atlas-based anatomical localization",
        "dimensions_method": "Axis-aligned voxel bounding box in image space; not clinical tumor diameters",
    }
    if not count:
        result.update({"bounding_box_voxels": None, "bounding_box_extent_mm": None,
                       "centroid_voxel": None, "centroid_world_ras_mm": None})
        return result
    points = np.argwhere(mask)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = points.mean(axis=0)
    world = affine @ np.r_[center, 1.0]
    result.update({
        "bounding_box_voxels": {"min": lower.tolist(), "max_inclusive": upper.tolist()},
        "bounding_box_extent_mm": [round(float(x), 2) for x in ((upper - lower + 1) * spacing)],
        "centroid_voxel": [round(float(x), 2) for x in center],
        "centroid_world_ras_mm": [round(float(x), 2) for x in world[:3]],
    })
    return result


def compare_masks(predicted, reference):
    predicted = np.asarray(predicted, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if predicted.shape != reference.shape:
        raise ValueError("Prediction and reference shapes differ")
    tp = int(np.count_nonzero(predicted & reference))
    fp = int(np.count_nonzero(predicted & ~reference))
    fn = int(np.count_nonzero(~predicted & reference))
    dice = 1.0 if not (tp + fp + fn) else 2 * tp / (2 * tp + fp + fn)
    precision = 1.0 if not (tp + fp) and not fn else tp / max(1, tp + fp)
    recall = 1.0 if not (tp + fn) and not fp else tp / max(1, tp + fn)
    return {"dice": round(dice, 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "true_positive_voxels": tp,
            "false_positive_voxels": fp, "false_negative_voxels": fn}

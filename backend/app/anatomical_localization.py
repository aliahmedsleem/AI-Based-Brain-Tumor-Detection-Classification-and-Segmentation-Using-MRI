"""Research-only localization from a registered, labeled anatomical atlas.

The atlas must be resampled into the MRI voxel grid by a separately validated
registration workflow. This module DOES NOT register or infer anatomy itself.
"""
import numpy as np

LOBES = {"frontal", "parietal", "temporal", "occipital", "insula", "cerebellum"}


def atlas_overlap(mask, atlas_labels, label_definitions, min_fraction=0.05):
    """Measure predicted-mask overlap with atlas regions; never infer from voxel position.

    label_definitions: {integer_label: {"lobe": "...", "hemisphere": "Left|Right|Midline|Unknown"}}
    """
    mask = np.asarray(mask, dtype=bool)
    atlas = np.asarray(atlas_labels)
    if mask.ndim != 3 or atlas.shape != mask.shape:
        raise ValueError("Atlas and prediction must share the same 3D voxel grid")
    if not np.isfinite(atlas).all():
        raise ValueError("Atlas contains invalid values")
    if not (0 <= min_fraction <= 1):
        raise ValueError("Invalid overlap cutoff")
    count = int(mask.sum())
    if not count:
        return {"status": "empty_prediction", "regions": [], "coverage_fraction": 0}
    results = []
    accounted = 0
    for label, details in label_definitions.items():
        code = int(label)
        if code == 0:
            continue
        lobe = details.get("lobe", "").lower()
        side = details.get("hemisphere", "")
        if lobe not in LOBES or side not in ("Left", "Right", "Midline", "Unknown"):
            raise ValueError(f"Invalid atlas region definition: {code}")
        voxels = int(np.count_nonzero(mask & (atlas == code)))
        accounted += voxels
        fraction = voxels / count
        if fraction >= min_fraction and voxels:
            results.append({"label": code, "lobe": lobe.title(),
                            "hemisphere": side, "overlap_voxels": voxels,
                            "fraction_of_predicted_mask": round(fraction, 4)})
    results.sort(key=lambda item: item["overlap_voxels"], reverse=True)
    coverage = accounted / count
    return {"status": "atlas_overlap_research_only", "regions": results,
            "coverage_fraction": round(coverage, 4),
            "method": "registered atlas label overlap with predicted whole-tumor mask",
            "warning": "Requires verified MRI-to-atlas registration and specialist confirmation; not a clinical diagnosis."}

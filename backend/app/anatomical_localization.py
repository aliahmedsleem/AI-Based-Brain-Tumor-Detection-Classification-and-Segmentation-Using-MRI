"""Research-only lobe localization from a registered anatomical label atlas.

The caller must establish image-to-atlas registration quality independently.
An atlas voxel grid/affine match alone does not validate registration.
"""
from collections import defaultdict

import numpy as np

LOBES = {"frontal", "parietal", "temporal", "occipital", "insula", "cerebellum"}
HEMISPHERES = {"Left", "Right", "Midline", "Unknown"}


def atlas_overlap(mask, atlas_labels, label_definitions, min_fraction=0.05):
    """Return per-label and aggregated lobe overlap fractions.

    All fractions are fractions of the *entire* predicted mask. The coverage
    fraction includes only valid, documented, mapped labels. Atlas label 0
    and unmapped regions are excluded. Region percentages need not total 100%.
    """
    mask = np.asarray(mask, dtype=bool)
    atlas = np.asarray(atlas_labels)
    if mask.ndim != 3 or atlas.shape != mask.shape:
        raise ValueError("Atlas and prediction must share the same 3D voxel grid")
    if not np.isfinite(atlas).all() or not np.all(atlas == np.rint(atlas)):
        raise ValueError("Atlas labels must be finite integers")
    if not (0 <= min_fraction <= 1):
        raise ValueError("Invalid overlap cutoff")
    if not isinstance(label_definitions, dict) or not label_definitions:
        raise ValueError("Atlas label definitions must be a nonempty mapping")

    normalized = {}
    for raw_code, details in label_definitions.items():
        code = int(raw_code)
        if code <= 0:
            raise ValueError("Atlas region labels must be positive integers")
        if code in normalized or not isinstance(details, dict):
            raise ValueError("Duplicate atlas code or invalid definition")
        lobe = str(details.get("lobe", "")).lower()
        hemisphere = details.get("hemisphere")
        if lobe not in LOBES or hemisphere not in HEMISPHERES:
            raise ValueError(f"Invalid atlas region definition: {code}")
        normalized[code] = (lobe.title(), hemisphere)

    count = int(mask.sum())
    if not count:
        return {"status": "empty_prediction", "regions": [], "lobes": [],
                "coverage_fraction": 0.0, "unmapped_fraction": 0.0,
                "dominant_lobe": None, "dominant_lobe_fraction": 0.0}

    codes, counts = np.unique(atlas[mask].astype(np.int64), return_counts=True)
    histogram = dict(zip(codes.tolist(), counts.tolist()))
    mapped_voxels = 0
    lobes = defaultdict(int)
    regions = []
    for code, (lobe, side) in normalized.items():
        voxels = int(histogram.get(code, 0))
        mapped_voxels += voxels
        lobes[(lobe, side)] += voxels
        fraction = voxels / count
        if voxels and fraction >= min_fraction:
            regions.append({"label": code, "lobe": lobe, "hemisphere": side,
                            "overlap_voxels": voxels,
                            "fraction_of_predicted_mask": round(fraction, 4)})
    regions.sort(key=lambda row: (-row["overlap_voxels"], row["label"]))
    aggregated = [
        {"lobe": lobe, "hemisphere": side, "overlap_voxels": voxels,
         "fraction_of_predicted_mask": round(voxels / count, 4)}
        for (lobe, side), voxels in lobes.items()
        if voxels and voxels / count >= min_fraction
    ]
    aggregated.sort(key=lambda row: (-row["overlap_voxels"], row["lobe"], row["hemisphere"]))
    # Only report a dominant lobe when enough of the mask is covered and
    # at least half the whole mask overlaps a single mapped lobe.
    coverage = mapped_voxels / count
    dominant = aggregated[0] if aggregated and coverage >= 0.8 and aggregated[0]["overlap_voxels"] / count >= 0.5 else None
    return {
        "status": "atlas_overlap_research_only",
        "regions": regions,
        "lobes": aggregated,
        "coverage_fraction": round(coverage, 4),
        "unmapped_fraction": round(1 - coverage, 4),
        "dominant_lobe": {"lobe": dominant["lobe"], "hemisphere": dominant["hemisphere"]} if dominant else None,
        "dominant_lobe_fraction": round(dominant["overlap_voxels"] / count, 4) if dominant else 0.0,
        "method": "registered atlas label overlap with predicted whole-tumor mask",
        "warning": "Atlas-derived research estimate only; registration and anatomical attribution require specialist review.",
    }

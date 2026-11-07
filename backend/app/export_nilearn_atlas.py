"""Export a Nilearn Harvard-Oxford cortical atlas and conservative lobe mapping.

Install: python -m pip install nilearn nibabel
Usage:
 python -m app.export_nilearn_atlas --data-dir C:\\NeuroScopeAI\\atlases --out-dir C:\\NeuroScopeAI\\atlases\\export
This is an atlas export, NOT an MRI registration or lobe diagnosis.
"""
import argparse
import json
from pathlib import Path

from app.harvard_oxford_mapping import REGIONS


def export(data_dir, out_dir, atlas_name="cort-maxprob-thr25-1mm"):
    from nilearn import datasets
    import nibabel as nib
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas = datasets.fetch_atlas_harvard_oxford(
        atlas_name, data_dir=str(data_dir), symmetric_split=False
    )
    # Nilearn versions may return a loaded Nifti image or a filesystem path.
    img = atlas.maps if isinstance(atlas.maps, nib.spatialimages.SpatialImage) else nib.load(str(atlas.maps))
    atlas_file = out_dir / "harvard_oxford_cortical_maxprob.nii.gz"
    nib.save(img, str(atlas_file))
    labels = [str(name) for name in atlas.labels]
    raw_codes = np.unique(np.asanyarray(img.dataobj))
    codes = raw_codes.astype(int)
    if not np.allclose(codes, raw_codes):
        raise ValueError("Atlas is not integer-labeled")
    if any(code < 0 or code >= len(labels) for code in codes):
        raise ValueError("Atlas label values do not match Nilearn label indices")
    mapping = {}
    excluded = []
    for code, name in enumerate(labels):
        if code == 0:
            continue
        normalized = " ".join(name.lower().split())
        matches = [lobe for lobe, words in REGIONS.items()
                   if any(word in normalized for word in words)]
        if len(matches) == 1:
            mapping[str(code)] = {"lobe": matches[0], "hemisphere": "Unknown"}
        else:
            excluded.append({"label": code, "name": name, "reason": "ambiguous or unmapped"})
    if not mapping:
        raise ValueError("No anatomical regions mapped")
    (out_dir / "atlas_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8")
    report = {
        "status": "atlas_export_only_requires_registration",
        "atlas_name": atlas_name,
        "atlas_nifti": str(atlas_file),
        "atlas_shape": list(img.shape),
        "labels_total": len(labels),
        "mapped": len(mapping),
        "excluded": excluded,
        "mapping_json": str(out_dir / "atlas_mapping.json"),
        "hemisphere": "Unknown",
        "note": "Nilearn maxprob label values directly index atlas.labels. Do not apply FSL XML offset here.",
        "next": "Obtain a compatible template MRI in the atlas space, then register to each subject and visually inspect all planes.",
    }
    (out_dir / "atlas_export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.data_dir, args.out_dir), indent=2))


if __name__ == "__main__":
    main()

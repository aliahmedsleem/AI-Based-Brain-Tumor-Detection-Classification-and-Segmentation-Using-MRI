"""Create a T1 MNI152 template resampled to the exported atlas grid.

This ensures matching template/atlas voxel geometry, NOT registration quality.
Install nilearn, nibabel, numpy.
Run: python -m app.prepare_mni_template --atlas PATH --out PATH
"""
import argparse
from pathlib import Path


def prepare(atlas_path, output_path):
    import nibabel as nib
    import numpy as np
    from nilearn import datasets, image

    atlas = nib.load(str(atlas_path))
    if atlas.ndim != 3:
        raise ValueError("Atlas must be 3D")
    # Nilearn's MNI152 template is T1-weighted. Use subject T1 for registration.
    template = datasets.load_mni152_template(resolution=1)
    aligned = image.resample_to_img(
        template, atlas, interpolation="continuous", force_resample=True,
        copy_header=True,
    )
    if aligned.shape != atlas.shape or not np.allclose(aligned.affine, atlas.affine, atol=1e-5):
        raise ValueError("Template resampling did not preserve atlas geometry")
    if not np.any(np.asanyarray(aligned.dataobj) > 0):
        raise ValueError("Template contains no nonzero tissue signal")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(aligned, str(output_path))
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print("T1 template in atlas grid:", prepare(args.atlas, args.out))
    print("Register against subject T1; visually inspect all three QA planes.")


if __name__ == "__main__":
    main()

"""Research-only affine atlas registration with optional tumor-exclusion mask.

Example:
python -m app.register_atlas_affine --subject subject_t1.nii.gz \
 --template mni152_t1_atlas_grid.nii.gz --labels atlas.nii.gz \
 --tumor-mask subject_seg.nii.gz --out-dir affine_run01
Visual specialist QA is mandatory; no registration is automatically verified.
"""
import argparse
import json
from pathlib import Path


def register(subject_path, template_path, labels_path, out_dir, tumor_mask_path=None):
    import SimpleITK as sitk
    import numpy as np
    from PIL import Image

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be empty; preserve previous QA runs")
    subject = sitk.ReadImage(str(subject_path), sitk.sitkFloat32)
    template = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
    labels = sitk.ReadImage(str(labels_path))
    if any(img.GetDimension() != 3 for img in (subject, template, labels)):
        raise ValueError("All images must be 3D")
    def same_grid(a, b):
        return (a.GetSize() == b.GetSize()
                and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-4)
                and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-3)
                and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5))
    if not same_grid(template, labels):
        raise ValueError("Template and atlas must share geometry")
    fixed = sitk.Normalize(subject)
    moving = sitk.Normalize(template)
    # Rigid initial estimate, followed by affine optimization.
    rigid = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    initial = sitk.AffineTransform(3)
    initial.SetCenter(rigid.GetCenter())
    initial.SetTranslation(rigid.GetTranslation())
    initial.SetMatrix(rigid.GetMatrix())

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=40)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.15, seed=42)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescentLineSearch(
        learningRate=0.5, numberOfIterations=250,
        convergenceMinimumValue=1e-6, convergenceWindowSize=12)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    excluded = 0
    if tumor_mask_path is not None:
        tumor = sitk.ReadImage(str(tumor_mask_path))
        if not same_grid(subject, tumor):
            raise ValueError("Tumor mask must match subject geometry exactly")
        # BraTS labels are 0,1,2,4; exclude tumor and a 5mm margin.
        binary = sitk.Cast(tumor > 0, sitk.sitkUInt8)
        radius = [max(1, int(np.ceil(5.0 / spacing))) for spacing in subject.GetSpacing()]
        dilated = sitk.BinaryDilate(binary, radius)
        brain = sitk.Cast(subject != 0, sitk.sitkUInt8)
        valid = sitk.Cast((brain > 0) & (dilated == 0), sitk.sitkUInt8)
        excluded = int(sitk.GetArrayViewFromImage(dilated).sum())
        if int(sitk.GetArrayViewFromImage(valid).sum()) < 10000:
            raise ValueError("Too few voxels remain after tumor exclusion")
        method.SetMetricFixedMask(valid)
    method.SetInitialTransform(initial, inPlace=False)
    transform = method.Execute(fixed, moving)
    aligned = sitk.Resample(template, subject, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    atlas = sitk.Resample(labels, subject, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    atlas_arr = sitk.GetArrayFromImage(atlas)
    if not np.any(atlas_arr):
        raise ValueError("No atlas labels intersect the subject")
    out.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(atlas, str(out / "registered_atlas.nii.gz"))
    sitk.WriteImage(aligned, str(out / "registered_template.nii.gz"))
    sitk.WriteTransform(transform, str(out / "template_to_subject.tfm"))
    fixed_arr = sitk.GetArrayFromImage(subject)
    moving_arr = sitk.GetArrayFromImage(aligned)

    def gray(a):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        nonzero = a[a > 0]
        if not nonzero.size:
            return np.zeros(a.shape, dtype=np.uint8)
        lo, hi = np.percentile(nonzero, [1, 99])
        return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

    qc = {}
    for plane, axis in (("axial", 0), ("coronal", 1), ("sagittal", 2)):
        index = fixed_arr.shape[axis] // 2
        a = gray(np.take(fixed_arr, index, axis=axis))
        b = gray(np.take(moving_arr, index, axis=axis))
        lab = np.take(atlas_arr, index, axis=axis)
        base = np.stack([a, a, a], axis=-1)
        overlay = base.copy()
        mask = lab > 0
        overlay[mask] = (0.7 * overlay[mask] + 0.3 * np.array([255, 70, 70])).astype(np.uint8)
        # Highlight atlas outer boundary in yellow for easier alignment review.
        from scipy.ndimage import binary_erosion
        boundary = mask & ~binary_erosion(mask)
        overlay[boundary] = [255, 220, 0]
        mosaic = np.concatenate((base, np.stack([b, b, b], axis=-1), overlay), axis=1)
        name = f"registration_qc_{plane}.png"
        Image.fromarray(mosaic).save(out / name)
        qc[plane] = {"file": name, "index": int(index),
                     "panels": ["subject", "registered_template", "atlas_overlay_with_boundary"]}
    report = {
        "status": "requires_visual_qa",
        "method": "multi-resolution affine mutual-information registration",
        "tumor_exclusion_used": tumor_mask_path is not None,
        "tumor_excluded_voxels_with_margin": excluded,
        "metric_value": float(method.GetMetricValue()),
        "optimizer_stop_condition": method.GetOptimizerStopConditionDescription(),
        "atlas_covered_voxels": int(np.count_nonzero(atlas_arr)),
        "output_atlas": "registered_atlas.nii.gz",
        "qc_images": qc,
        "limitations": [
            "Affine registration cannot model local deformation or tumor mass effect.",
            "Tumor exclusion is based on a provided mask, not a clinical validation.",
            "Inspect all three QA planes and region boundaries; do not attest registration automatically.",
            "Research use only; specialist anatomical review required before lobe attribution."
        ],
    }
    (out / "registration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--tumor-mask", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(register(args.subject, args.template, args.labels, args.out_dir,
                              args.tumor_mask), indent=2))


if __name__ == "__main__":
    main()

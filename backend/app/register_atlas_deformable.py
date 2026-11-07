"""Experimental affine-initialized B-spline atlas registration; NEVER auto-verify anatomy.

Uses the prior subject-to-template affine transform as a fixed moving transform,
and optimizes a low-resolution B-spline in subject space. A tumor exclusion mask
is optional and should only be used when available independently.
"""
import argparse
import json
from pathlib import Path


def run(subject_path, template_path, labels_path, affine_path, out_dir,
        tumor_mask_path=None, grid_mm=60.0):
    import SimpleITK as sitk
    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_erosion

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; retain prior QA evidence")
    if grid_mm < 30:
        raise ValueError("B-spline grid must be >=30 mm to limit overfitting")
    subject = sitk.ReadImage(str(subject_path), sitk.sitkFloat32)
    template = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
    labels = sitk.ReadImage(str(labels_path))
    if any(x.GetDimension() != 3 for x in (subject, template, labels)):
        raise ValueError("All inputs must be 3D")
    def same_grid(a, b):
        return (a.GetSize() == b.GetSize()
                and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-4)
                and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-3)
                and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5))
    if not same_grid(template, labels):
        raise ValueError("Template and atlas label geometry differ")
    affine = sitk.ReadTransform(str(affine_path))
    if affine.GetDimension() != 3:
        raise ValueError("Affine transform must be 3D")
    # SimpleITK resampling transforms map fixed/subject physical coordinates to moving/template.
    physical_size = [(n - 1) * s for n, s in zip(subject.GetSize(), subject.GetSpacing())]
    mesh = [max(1, int(round(v / grid_mm))) for v in physical_size]
    bspline = sitk.BSplineTransformInitializer(subject, mesh, order=3)
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=40)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.12, seed=42)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                                numberOfIterations=80, maximumNumberOfCorrections=5,
                                maximumNumberOfFunctionEvaluations=400, costFunctionConvergenceFactor=1e7)
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    excluded = 0
    if tumor_mask_path:
        tumor = sitk.ReadImage(str(tumor_mask_path))
        if not same_grid(subject, tumor):
            raise ValueError("Tumor mask and subject geometry differ")
        radius = [max(1, int(np.ceil(5 / s))) for s in subject.GetSpacing()]
        expanded = sitk.BinaryDilate(sitk.Cast(tumor > 0, sitk.sitkUInt8), radius)
        valid = sitk.Cast((subject != 0) & (expanded == 0), sitk.sitkUInt8)
        excluded = int(np.count_nonzero(sitk.GetArrayViewFromImage(expanded)))
        if np.count_nonzero(sitk.GetArrayViewFromImage(valid)) < 10000:
            raise ValueError("Too few tissue voxels remain after masking")
        method.SetMetricFixedMask(valid)
    method.SetMovingInitialTransform(affine)
    method.SetInitialTransform(bspline, inPlace=False)
    fitted = method.Execute(sitk.Normalize(subject), sitk.Normalize(template))
    # Composite evaluates last-added first: affine(bspline(subject-coordinate)).
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(affine)
    composite.AddTransform(fitted)
    displacement = sitk.TransformToDisplacementField(
        composite, sitk.sitkVectorFloat64, subject.GetSize(),
        subject.GetOrigin(), subject.GetSpacing(), subject.GetDirection())
    jacobian = sitk.DisplacementFieldJacobianDeterminant(displacement)
    jac = sitk.GetArrayFromImage(jacobian)
    tissue = sitk.GetArrayFromImage(subject) > 0
    values = jac[tissue & np.isfinite(jac)]
    if not values.size:
        raise ValueError("No finite Jacobian values in tissue")
    folded = int(np.count_nonzero(values <= 0))
    aligned = sitk.Resample(template, subject, composite, sitk.sitkLinear, 0, sitk.sitkFloat32)
    atlas = sitk.Resample(labels, subject, composite, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    atlas_arr = sitk.GetArrayFromImage(atlas)
    if not np.any(atlas_arr):
        raise ValueError("Atlas has no coverage after registration")
    out.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(aligned, str(out / "registered_template.nii.gz"))
    sitk.WriteImage(atlas, str(out / "registered_atlas.nii.gz"))
    sitk.WriteImage(jacobian, str(out / "jacobian_determinant.nii.gz"))
    # Flatten nested CompositeTransforms before HDF5 serialization.
    serializable = sitk.CompositeTransform(composite)
    serializable.FlattenTransform()
    sitk.WriteTransform(serializable, str(out / "template_to_subject_composite.h5"))
    fixed_arr = sitk.GetArrayFromImage(subject)
    moving_arr = sitk.GetArrayFromImage(aligned)
    def gray(a):
        a = np.nan_to_num(a)
        positive = a[a > 0]
        if not positive.size:
            return np.zeros(a.shape, dtype=np.uint8)
        lo, hi = np.percentile(positive, [1, 99])
        return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    qc = {}
    for plane, axis in (("axial", 0), ("coronal", 1), ("sagittal", 2)):
        idx = fixed_arr.shape[axis] // 2
        a = gray(np.take(fixed_arr, idx, axis=axis))
        b = gray(np.take(moving_arr, idx, axis=axis))
        mask = np.take(atlas_arr, idx, axis=axis) > 0
        base = np.stack((a, a, a), axis=-1)
        overlay = base.copy()
        overlay[mask] = (0.7 * overlay[mask] + 0.3 * np.array([255, 70, 70])).astype(np.uint8)
        overlay[mask & ~binary_erosion(mask)] = [255, 220, 0]
        name = f"registration_qc_{plane}.png"
        Image.fromarray(np.concatenate((base, np.stack((b, b, b), axis=-1), overlay), axis=1)).save(out / name)
        qc[plane] = {"file": name, "index": int(idx)}
    report = {
        "status": "requires_visual_qa",
        "method": "affine-initialized B-spline mutual-information registration",
        "grid_mm": grid_mm, "mesh_size": mesh,
        "tumor_exclusion_used": bool(tumor_mask_path),
        "tumor_excluded_voxels_with_margin": excluded,
        "optimizer_stop_condition": method.GetOptimizerStopConditionDescription(),
        "metric_value": float(method.GetMetricValue()),
        "jacobian_min": float(values.min()),
        "jacobian_p01": float(np.percentile(values, 1)),
        "jacobian_p50": float(np.median(values)),
        "jacobian_p99": float(np.percentile(values, 99)),
        "jacobian_nonpositive_tissue_voxels": folded,
        "jacobian_nonpositive_tissue_fraction": float(folded / len(values)),
        "atlas_covered_voxels": int(np.count_nonzero(atlas_arr)),
        "qc_images": qc,
        "output_atlas": "registered_atlas.nii.gz",
        "limitations": [
            "Jacobian checks detect folding but do not establish anatomical accuracy.",
            "Mutual information values from different runs are not directly comparable.",
            "B-spline can distort anatomy near tumors; review atlas region boundaries, not just its outer silhouette.",
            "Research-only: specialist review required; never mark registration verified automatically."
        ],
    }
    (out / "registration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--affine-transform", type=Path, required=True)
    p.add_argument("--tumor-mask", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--grid-mm", type=float, default=60)
    a = p.parse_args()
    print(json.dumps(run(a.subject, a.template, a.labels, a.affine_transform,
                         a.out_dir, a.tumor_mask, a.grid_mm), indent=2))


if __name__ == "__main__":
    main()

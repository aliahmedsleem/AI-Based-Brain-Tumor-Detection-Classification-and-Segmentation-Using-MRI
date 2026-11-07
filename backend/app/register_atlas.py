"""Register a template MRI and labeled atlas to a subject MRI for RESEARCH QA.

Install optional dependency: python -m pip install SimpleITK
Inputs must be preprocessed compatible MRIs (same contrast, skull-stripping
preferred). Registration does not establish anatomical correctness.

Example:
python -m app.register_atlas --subject case_flair.nii.gz --template template_flair.nii.gz \\
  --labels atlas_lobes.nii.gz --out-dir atlas_registration
"""
import argparse
import json
from pathlib import Path


def register(subject_path, template_path, labels_path, out_dir):
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK is required: python -m pip install SimpleITK") from exc
    import numpy as np
    from PIL import Image

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Output directory must be empty to prevent overwriting QA evidence")
    subject = sitk.ReadImage(str(subject_path), sitk.sitkFloat32)
    template = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
    labels = sitk.ReadImage(str(labels_path))
    if subject.GetDimension() != 3 or template.GetDimension() != 3 or labels.GetDimension() != 3:
        raise ValueError("Only 3D NIfTI images supported")
    if template.GetSize() != labels.GetSize() or not np.allclose(template.GetDirection(), labels.GetDirection(), atol=1e-5) or not np.allclose(template.GetOrigin(), labels.GetOrigin(), atol=1e-3) or not np.allclose(template.GetSpacing(), labels.GetSpacing(), atol=1e-4):
        raise ValueError("Atlas labels and template MRI must share physical geometry")
    fixed = sitk.Normalize(subject)
    moving = sitk.Normalize(template)
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=0.001, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=False)
    transform = registration.Execute(fixed, moving)
    aligned = sitk.Resample(template, subject, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    atlas = sitk.Resample(labels, subject, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    arr = sitk.GetArrayViewFromImage(atlas)
    if not np.count_nonzero(arr):
        raise ValueError("Registration produced no atlas coverage")
    out_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(atlas, str(out_dir / "registered_atlas.nii.gz"))
    sitk.WriteImage(aligned, str(out_dir / "registered_template.nii.gz"))
    sitk.WriteTransform(transform, str(out_dir / "template_to_subject.tfm"))
    # Generate independently inspectable orthogonal QA panels.
    fixed_arr = sitk.GetArrayFromImage(subject)
    moving_arr = sitk.GetArrayFromImage(aligned)
    atlas_arr = sitk.GetArrayFromImage(atlas)

    def normalize(slice_2d):
        finite = slice_2d[np.isfinite(slice_2d) & (slice_2d != 0)]
        if not finite.size:
            return np.zeros_like(slice_2d, dtype=np.uint8)
        low, high = np.percentile(finite, [1, 99])
        return (np.clip((slice_2d - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)

    qc_images = {}
    for plane, axis in (("axial", 0), ("coronal", 1), ("sagittal", 2)):
        index = fixed_arr.shape[axis] // 2
        fixed_slice = np.take(fixed_arr, index, axis=axis)
        moving_slice = np.take(moving_arr, index, axis=axis)
        labels_slice = np.take(atlas_arr, index, axis=axis)
        fixed_gray = normalize(fixed_slice)
        moving_gray = normalize(moving_slice)
        overlay = np.stack([fixed_gray] * 3, axis=-1)
        inside = labels_slice > 0
        overlay[inside] = (
            overlay[inside].astype(np.float32) * 0.65 +
            np.array([255, 60, 60], dtype=np.float32) * 0.35
        ).astype(np.uint8)
        panels = [
            np.stack([fixed_gray] * 3, axis=-1),
            np.stack([moving_gray] * 3, axis=-1),
            overlay,
        ]
        mosaic = np.concatenate(panels, axis=1)
        filename = f"registration_qc_{plane}.png"
        Image.fromarray(mosaic).save(out_dir / filename)
        qc_images[plane] = {"file": filename, "index": int(index),
                            "panels": ["subject", "registered_template", "atlas_overlay"]}
    report = {
        "status": "requires_visual_qa", "method": "multi-resolution rigid Euler3D mutual-information registration",
        "metric_value": float(registration.GetMetricValue()),
        "optimizer_stop_condition": registration.GetOptimizerStopConditionDescription(),
        "atlas_covered_voxels": int(np.count_nonzero(arr)),
        "qc_image": "registration_qc_axial.png",
        "qc_images": qc_images,
        "output_atlas": "registered_atlas.nii.gz",
        "limitations": [
            "Rigid alignment cannot correct all individual anatomical variation or mass effect.",
            "Registration metric and orthogonal QC snapshots do not prove anatomical accuracy.",
            "A trained specialist must review all three planes and atlas labels before reporting a lobe.",
            "Use a modality-compatible template MRI and corresponding label atlas.",
        ],
    }
    (out_dir / "registration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = register(args.subject, args.template, args.labels, args.out_dir)
    print(json.dumps(result, indent=2))
    print("Review registration in all planes before uploading registered_atlas.nii.gz to NeuroScope.")


if __name__ == "__main__":
    main()

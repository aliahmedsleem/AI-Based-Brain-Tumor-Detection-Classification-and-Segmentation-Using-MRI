"""Offline validation-only visual inspection of binary whole-tumor segmentation.

Uses the original saved split.json, NEVER the held-out test partition.
Outputs per-case metrics, NIfTI prediction and PNG overlays.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="Existing training run containing split.json")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--max-cases", type=int, default=7)
    args = p.parse_args()
    if args.max_cases < 1:
        p.error("--max-cases must be positive")
    if args.out.resolve().is_relative_to(args.run.resolve()):
        p.error("Use an output directory outside the training run to protect checkpoints")

    import numpy as np
    import nibabel as nib
    import torch
    from PIL import Image
    from monai.networks.nets import UNet
    from monai.inferers import sliding_window_inference

    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable")
    split = json.loads((args.run / "split.json").read_text(encoding="utf-8"))
    records = split["val"][:args.max_cases]
    if not records:
        p.error("No validation records in split.json")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("task") != "binary_whole_tumor":
        p.error("Only binary whole-tumor checkpoints are supported")
    roi = int(checkpoint["roi_size"])
    model = UNet(spatial_dims=3, in_channels=1, out_channels=1,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                 num_res_units=2)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = []

    for record in records:
        name = record["patient_id"]
        img = nib.as_closest_canonical(nib.load(record["image"], mmap=False))
        seg = nib.as_closest_canonical(nib.load(record["label"], mmap=False))
        if len(img.shape) != 3 or img.shape != seg.shape or not np.allclose(img.affine, seg.affine, rtol=1e-4, atol=1e-3):
            raise ValueError(f"Image and label geometry mismatch: {name}")
        volume = img.get_fdata(dtype=np.float32, caching="unchanged")
        truth = seg.get_fdata(dtype=np.float32, caching="unchanged") > 0
        if not np.isfinite(volume).all() or not np.isfinite(truth).all():
            raise ValueError(f"Non-finite input: {name}")
        foreground = volume != 0
        if not foreground.any():
            raise ValueError(f"Empty MRI foreground: {name}")
        values = volume[foreground]
        lo, hi = np.percentile(values, [0.5, 99.5])
        values = np.clip(values, lo, hi)
        normalized = np.zeros_like(volume, dtype=np.float32)
        normalized[foreground] = (values - values.mean()) / max(float(values.std()), 1e-6)
        with torch.inference_mode():
            logits = sliding_window_inference(
                torch.from_numpy(normalized[None, None]).to(args.device),
                roi_size=(roi, roi, roi), sw_batch_size=1, predictor=model, overlap=0.25)
            pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5)
        tp = int(np.logical_and(pred, truth).sum())
        fp = int(np.logical_and(pred, ~truth).sum())
        fn = int(np.logical_and(~pred, truth).sum())
        dice = (2 * tp) / max(1, 2 * tp + fp + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        case_out = args.out / name
        case_out.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(pred.astype(np.uint8), img.affine, img.header), str(case_out / "prediction.nii.gz"))

        # Select axial slice with largest true tumor area; visual inspection only.
        z = int(np.argmax(truth.sum(axis=(0, 1))))
        slice_img = normalized[:, :, z]
        gray = np.uint8(np.clip((slice_img + 2.5) / 5 * 255, 0, 255))
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        gt_rgb = rgb.copy()
        gt_rgb[truth[:, :, z]] = (30, 220, 90)
        pred_rgb = rgb.copy()
        pred_rgb[pred[:, :, z]] = (240, 75, 70)
        combined = np.concatenate((rgb, gt_rgb, pred_rgb), axis=1)
        Image.fromarray(np.rot90(combined)).save(case_out / "comparison.png")
        metrics.append({"patient_id": name, "dice": round(dice, 5),
                        "precision": round(precision, 5), "recall": round(recall, 5),
                        "tp": tp, "fp": fp, "fn": fn, "axial_slice": z})
        print(f"{name}: Dice={dice:.4f} Precision={precision:.4f} Recall={recall:.4f}", flush=True)
    report = {"partition": "validation_only", "checkpoint": str(args.checkpoint),
              "cases": metrics, "mean_dice": float(np.mean([x["dice"] for x in metrics])),
              "note": "Research only. PNG panels: MRI | true tumor (green) | predicted tumor (red)."}
    (args.out / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved validation report and images to {args.out}")


if __name__ == "__main__":
    main()

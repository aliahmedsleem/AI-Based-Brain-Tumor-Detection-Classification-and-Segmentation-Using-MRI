"""Validation-only probability threshold analysis; never touches the test split.

Reuses trained binary FLAIR 3D UNet to compare decision thresholds. Results are
exploratory and must not be reported as independent test performance.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])
    args = parser.parse_args()
    if not args.thresholds or any(not 0 < t < 1 for t in args.thresholds):
        parser.error("Each threshold must be strictly between zero and one")
    if args.out.resolve().is_relative_to(args.run.resolve()):
        parser.error("Output must be outside the training run")

    import numpy as np
    import nibabel as nib
    import torch
    from monai.networks.nets import UNet
    from monai.inferers import sliding_window_inference

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    split = json.loads((args.run / "split.json").read_text(encoding="utf-8"))
    records = split["val"]
    if not records:
        parser.error("No validation cases")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("task") != "binary_whole_tumor":
        parser.error("Checkpoint task mismatch")
    roi = int(checkpoint["roi_size"])
    model = UNet(spatial_dims=3, in_channels=1, out_channels=1,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                 num_res_units=2)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for record in records:
        name = record["patient_id"]
        img = nib.as_closest_canonical(nib.load(record["image"], mmap=False))
        seg = nib.as_closest_canonical(nib.load(record["label"], mmap=False))
        if len(img.shape) != 3 or img.shape != seg.shape or not np.allclose(img.affine, seg.affine, rtol=1e-4, atol=1e-3):
            raise ValueError(f"Geometry mismatch: {name}")
        volume = img.get_fdata(dtype=np.float32, caching="unchanged")
        truth = seg.get_fdata(dtype=np.float32, caching="unchanged") > 0
        if not np.isfinite(volume).all():
            raise ValueError(f"Nonfinite MRI: {name}")
        foreground = volume != 0
        if not foreground.any():
            raise ValueError(f"Empty foreground: {name}")
        values = volume[foreground]
        lo, hi = np.percentile(values, [0.5, 99.5])
        values = np.clip(values, lo, hi)
        normalized = np.zeros_like(volume, dtype=np.float32)
        normalized[foreground] = (values - values.mean()) / max(float(values.std()), 1e-6)
        with torch.inference_mode():
            logits = sliding_window_inference(
                torch.from_numpy(normalized[None, None]).to(args.device),
                roi_size=(roi, roi, roi), sw_batch_size=1, predictor=model, overlap=0.25)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        for threshold in args.thresholds:
            pred = probs > threshold
            tp = int(np.count_nonzero(pred & truth))
            fp = int(np.count_nonzero(pred & ~truth))
            fn = int(np.count_nonzero(~pred & truth))
            rows.append({"patient_id": name, "threshold": threshold,
                         "dice": 2 * tp / max(1, 2 * tp + fp + fn),
                         "precision": tp / max(1, tp + fp),
                         "recall": tp / max(1, tp + fn),
                         "false_positive_voxels": fp})
        print(f"Processed {name}", flush=True)

    summary = []
    for threshold in args.thresholds:
        subset = [row for row in rows if row["threshold"] == threshold]
        summary.append({"threshold": threshold, "mean_dice": float(np.mean([r["dice"] for r in subset])),
                        "mean_precision": float(np.mean([r["precision"] for r in subset])),
                        "mean_recall": float(np.mean([r["recall"] for r in subset]))})
    report = {"partition": "validation_only", "per_case": rows, "summary": summary,
              "note": "Threshold exploration on validation cases only; no independent test evaluation."}
    (args.out / "threshold_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in summary:
        print(f"threshold={row['threshold']:.2f} dice={row['mean_dice']:.4f} precision={row['mean_precision']:.4f} recall={row['mean_recall']:.4f}", flush=True)
    print(f"Report: {args.out / 'threshold_report.json'}")


if __name__ == "__main__":
    main()

"""Compare largest-connected-component postprocessing on validation only.

Requires scipy in the isolated training environment. No test data is accessed.
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
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("Threshold must be between zero and one")
    if args.out.resolve().is_relative_to(args.run.resolve()):
        parser.error("Output must be outside training run")
    import numpy as np
    import nibabel as nib
    import torch
    from scipy.ndimage import label as connected_label
    from monai.networks.nets import UNet
    from monai.inferers import sliding_window_inference

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    records = json.loads((args.run / "split.json").read_text(encoding="utf-8"))["val"]
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

    def metrics(pred, truth):
        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = int(np.count_nonzero(~pred & truth))
        return {"dice": 2 * tp / max(1, 2 * tp + fp + fn),
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "false_positive_voxels": fp}

    for record in records:
        name = record["patient_id"]
        img = nib.as_closest_canonical(nib.load(record["image"], mmap=False))
        seg = nib.as_closest_canonical(nib.load(record["label"], mmap=False))
        if len(img.shape) != 3 or img.shape != seg.shape or not np.allclose(img.affine, seg.affine, rtol=1e-4, atol=1e-3):
            raise ValueError(f"Geometry mismatch: {name}")
        volume = img.get_fdata(dtype=np.float32, caching="unchanged")
        labels = seg.get_fdata(dtype=np.float32, caching="unchanged")
        if not np.isfinite(volume).all() or not np.isfinite(labels).all():
            raise ValueError(f"Nonfinite input: {name}")
        truth = labels > 0
        foreground = volume != 0
        if not foreground.any():
            raise ValueError(f"Empty MRI foreground: {name}")
        values = volume[foreground]
        lo, hi = np.percentile(values, [0.5, 99.5])
        clipped = np.clip(values, lo, hi)
        normalized = np.zeros_like(volume, dtype=np.float32)
        normalized[foreground] = (clipped - clipped.mean()) / max(float(clipped.std()), 1e-6)
        with torch.inference_mode():
            logits = sliding_window_inference(
                torch.from_numpy(normalized[None, None]).to(args.device),
                roi_size=(roi, roi, roi), sw_batch_size=1, predictor=model, overlap=0.25)
            pred = torch.sigmoid(logits)[0, 0].cpu().numpy() > args.threshold
        cc, count = connected_label(pred)
        if count:
            sizes = np.bincount(cc.ravel())
            sizes[0] = 0
            cleaned = cc == int(np.argmax(sizes))
        else:
            cleaned = pred.copy()
        raw_metrics, clean_metrics = metrics(pred, truth), metrics(cleaned, truth)
        rows.append({"patient_id": name, "components": int(count),
                     "raw": raw_metrics, "largest_component": clean_metrics})
        print(f"{name}: raw={raw_metrics['dice']:.4f} largest_component={clean_metrics['dice']:.4f} components={count}", flush=True)

    summary = {}
    for key in ("raw", "largest_component"):
        summary[key] = {metric: float(np.mean([row[key][metric] for row in rows]))
                        for metric in ("dice", "precision", "recall")}
    report = {"partition": "validation_only", "threshold": args.threshold,
              "postprocess": "largest 3D connected component (6-connectivity)",
              "summary": summary, "cases": rows,
              "warning": "Postprocessing selected on validation only; test not evaluated."}
    path = args.out / "postprocess_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, values in summary.items():
        print(f"{key}: Dice={values['dice']:.4f} Precision={values['precision']:.4f} Recall={values['recall']:.4f}")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()

"""One-time held-out test evaluation for frozen binary FLAIR 3D UNet.

Run only after threshold and postprocessing are finalized using validation.
Never use this report to retune hyperparameters.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--postprocess", choices=("largest", "none"), default="largest")
    args = p.parse_args()
    if not 0 < args.threshold < 1:
        p.error("Invalid threshold")
    if args.out.resolve().is_relative_to(args.run.resolve()):
        p.error("Output must be outside the training run")
    if args.out.exists() and any(args.out.iterdir()):
        p.error("Output folder must be new or empty; refusing to overwrite a test report")

    import numpy as np
    import nibabel as nib
    import torch
    from scipy.ndimage import label as connected_label
    from monai.networks.nets import UNet
    from monai.inferers import sliding_window_inference

    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    split = json.loads((args.run / "split.json").read_text(encoding="utf-8"))
    records = split["test"]
    if not records:
        p.error("No test records")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("task") != "binary_whole_tumor":
        p.error("Checkpoint task mismatch")
    roi = int(ckpt["roi_size"])
    model = UNet(spatial_dims=3, in_channels=1, out_channels=1,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                 num_res_units=2)
    model.load_state_dict(ckpt["state_dict"], strict=True)
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
        label = seg.get_fdata(dtype=np.float32, caching="unchanged")
        if not np.isfinite(volume).all() or not np.isfinite(label).all():
            raise ValueError(f"Nonfinite input: {name}")
        truth = label > 0
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
        if args.postprocess == "largest":
            components, count = connected_label(pred)
            if count:
                sizes = np.bincount(components.ravel())
                sizes[0] = 0
                pred = components == int(np.argmax(sizes))
        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = int(np.count_nonzero(~pred & truth))
        row = {"patient_id": name, "dice": 2 * tp / max(1, 2 * tp + fp + fn),
               "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
               "tp": tp, "fp": fp, "fn": fn}
        rows.append(row)
        print(f"{name}: Dice={row['dice']:.4f} Precision={row['precision']:.4f} Recall={row['recall']:.4f}", flush=True)
    summary = {metric: float(np.mean([row[metric] for row in rows]))
               for metric in ("dice", "precision", "recall")}
    report = {"partition": "held_out_test", "threshold": args.threshold,
              "postprocess": args.postprocess, "checkpoint": str(args.checkpoint.resolve()),
              "cases": rows, "summary": summary,
              "note": "One-time evaluation of validation-selected configuration; research use only."}
    path = args.out / "test_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("HELD-OUT TEST RESULTS (do not tune against these):")
    print(f"Dice={summary['dice']:.4f} Precision={summary['precision']:.4f} Recall={summary['recall']:.4f}")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()

"""Offline experimental binary whole-tumor 3D UNet training; never run in web server."""
import argparse
import json
import random
from pathlib import Path

from training.manifest import load_manifest, split_patients


def train(args):
    import numpy as np
    import nibabel as nib
    import torch
    from monai.networks.nets import UNet
    from monai.losses import DiceCELoss
    from monai.inferers import sliding_window_inference
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    partitions = split_patients(load_manifest(args.manifest), args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split.json").write_text(json.dumps(partitions, indent=2), encoding="utf-8")
    (args.out / "config.json").write_text(json.dumps({
        "seed": args.seed, "epochs": args.epochs, "roi_size": args.roi_size,
        "learning_rate": args.lr, "task": "binary_whole_tumor",
        "model": "MONAI 3D UNet", "device": str(device),
    }, indent=2), encoding="utf-8")

    def read_case(record):
        img = nib.as_closest_canonical(nib.load(record["image"], mmap=False))
        seg = nib.as_closest_canonical(nib.load(record["label"], mmap=False))
        if len(img.shape) != 3 or len(seg.shape) != 3 or img.shape != seg.shape:
            raise ValueError("Expected aligned 3D image and label for " + record["patient_id"])
        if not np.allclose(img.affine, seg.affine, rtol=1e-4, atol=1e-3):
            raise ValueError("Image/label affine mismatch for " + record["patient_id"])
        image = img.get_fdata(dtype=np.float32, caching="unchanged")
        labels = seg.get_fdata(dtype=np.float32, caching="unchanged")
        if not np.isfinite(image).all() or not np.isfinite(labels).all():
            raise ValueError("Non-finite image or label values")
        mask = image != 0
        if not mask.any():
            raise ValueError("Empty foreground image")
        values = image[mask]
        lo, hi = np.percentile(values, [0.5, 99.5])
        clipped = np.clip(values, lo, hi)
        std = float(clipped.std())
        image = np.zeros_like(image, dtype=np.float32)
        image[mask] = (clipped - clipped.mean()) / max(std, 1e-6)
        return torch.from_numpy(image[None].copy()), torch.from_numpy((labels[None] > 0).astype(np.float32))

    def patch_pair(image, label):
        target = (args.roi_size,) * 3
        pads = []
        for actual in reversed(image.shape[1:]):
            pads.extend((0, max(0, args.roi_size - actual)))
        image, label = F.pad(image, pads), F.pad(label, pads)
        # Favor tumor-containing patches for highly imbalanced segmentation data.
        positive = torch.nonzero(label[0] > 0, as_tuple=False)
        if len(positive) and random.random() < 0.7:
            center = positive[random.randrange(len(positive))].tolist()
            starts = [max(0, min(d - size, int(c) - size // 2))
                      for d, size, c in zip(image.shape[1:], target, center)]
        else:
            starts = [random.randint(0, d - size) for d, size in zip(image.shape[1:], target)]
        sl = tuple(slice(start, start + size) for start, size in zip(starts, target))
        return image[(slice(None),) + sl].unsqueeze(0), label[(slice(None),) + sl].unsqueeze(0)

    model = UNet(spatial_dims=3, in_channels=1, out_channels=1,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                 num_res_units=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=False)
    best_dice = -1.0

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for record in partitions["train"]:
            x, y = read_case(record)
            x, y = patch_pair(x, y)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        dice_scores = []
        with torch.no_grad():
            for record in partitions["val"]:
                x, y = read_case(record)
                prediction = sliding_window_inference(x.unsqueeze(0).to(device),
                    roi_size=(args.roi_size,) * 3, sw_batch_size=1,
                    predictor=model, overlap=0.25)
                pred = (torch.sigmoid(prediction).cpu() > 0.5).float()
                target = y.unsqueeze(0)
                intersection = (pred * target).sum().item()
                denominator = pred.sum().item() + target.sum().item()
                dice_scores.append((2 * intersection + 1e-6) / (denominator + 1e-6))
        score = float(np.mean(dice_scores))
        print(f"epoch={epoch+1}/{args.epochs} train_loss={np.mean(train_losses):.4f} val_dice={score:.4f}", flush=True)
        if score > best_dice:
            best_dice = score
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1,
                        "val_dice": score, "task": "binary_whole_tumor",
                        "roi_size": args.roi_size}, args.out / "best_val.pt")
    print(f"Finished. Best validation Dice={best_dice:.4f}; test set NOT evaluated.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--roi-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.roi_size < 32 or args.roi_size % 16 or not 0 < args.lr < 1:
        parser.error("epochs>=1, roi-size>=32 divisible by 16, and 0<lr<1 required")
    train(args)


if __name__ == "__main__":
    main()

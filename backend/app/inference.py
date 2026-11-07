"""Optional local-only research inference; never auto-download model weights."""
import os
from pathlib import Path


def configured_checkpoint():
    raw = os.environ.get("NEUROSCOPE_MODEL_PATH", "").strip()
    return Path(raw) if raw else None


def infer(volume):
    import numpy as np
    import torch
    from monai.networks.nets import UNet
    from monai.inferers import sliding_window_inference
    from scipy.ndimage import label as connected_label

    path = configured_checkpoint()
    if path is None or not path.is_file():
        raise RuntimeError("Set NEUROSCOPE_MODEL_PATH to a trusted local best_val.pt checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the API Python environment; use the isolated training Python to run the API")
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("Expected finite 3D FLAIR NIfTI volume")
    foreground = volume != 0
    if not foreground.any():
        raise ValueError("MRI has no nonzero foreground")
    vals = volume[foreground]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    vals = np.clip(vals, lo, hi)
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[foreground] = (vals - vals.mean()) / max(float(vals.std()), 1e-6)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("task") != "binary_whole_tumor":
        raise ValueError("Unsupported checkpoint task")
    roi = int(checkpoint["roi_size"])
    model = UNet(spatial_dims=3, in_channels=1, out_channels=1,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.cuda().eval()
    try:
        with torch.inference_mode():
            logits = sliding_window_inference(
                torch.from_numpy(normalized[None, None]).cuda(),
                roi_size=(roi, roi, roi), sw_batch_size=1, predictor=model, overlap=0.25)
            mask = torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.90
    finally:
        model.cpu()
        del model
    components, count = connected_label(mask)
    if count:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        mask = components == int(np.argmax(sizes))
    return mask.astype(np.uint8)


def overlay_png(volume, mask, axis, index, lower, upper, opacity=0.55):
    import base64
    from io import BytesIO
    import numpy as np
    from PIL import Image

    opacity = max(0.0, min(1.0, float(opacity)))
    gray = np.rot90(np.take(volume, index, axis=axis))
    marked = np.rot90(np.take(mask, index, axis=axis)).astype(bool)
    pixels = (np.clip((gray - lower) / max(upper - lower, 1e-6), 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(pixels[:, :, None], 3, axis=2)
    rgb[marked] = (rgb[marked].astype(np.float32) * (1 - opacity) + np.array([255, 65, 65]) * opacity).astype(np.uint8)
    picture = Image.fromarray(rgb, mode="RGB")
    picture.thumbnail((512, 512))
    buf = BytesIO()
    picture.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

from pathlib import Path
from io import BytesIO
import base64
import tempfile
import secrets
import time
from threading import RLock
from fastapi import Query

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NeuroScope AI Research API", version="0.9.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    from app.inference import configured_checkpoint
    path = configured_checkpoint()
    return {"status": "ok", "mode": "research", "model_configured": bool(path and path.is_file())}


def preview_image(volume, axis, index, lower, upper):
    import numpy as np
    from PIL import Image

    slice_2d = np.take(volume, index, axis=axis)
    slice_2d = np.rot90(slice_2d)
    normalized = np.clip((slice_2d - lower) / (upper - lower), 0, 1)
    pixels = (normalized * 255).astype(np.uint8)
    picture = Image.fromarray(pixels, mode="L")
    picture.thumbnail((512, 512))
    buffer = BytesIO()
    picture.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")



# Research prototype: one active volume in memory, no disk persistence.
# Deploy only on trusted localhost; session IDs are not user authentication.
CACHE_TTL_SECONDS = 600
CACHE_MAX_BYTES = 256 * 1024 * 1024
_cache = {}
_cache_lock = RLock()


def _get_session(study_id: str):
    with _cache_lock:
        now = time.monotonic()
        for key, item in list(_cache.items()):
            if now - item["last_access"] > CACHE_TTL_SECONDS:
                del _cache[key]
        item = _cache.get(study_id)
        if item is None:
            raise HTTPException(status_code=404, detail="MRI session expired. Inspect the file again.")
        item["last_access"] = now
        return item


@app.post("/api/studies/inspect")
async def inspect_study(file: UploadFile = File(...), volume_index: int = Query(default=0, ge=0)):
    if not file.filename or not file.filename.lower().endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="Only .nii or .nii.gz supported")
    import nibabel as nib
    import numpy as np

    with tempfile.TemporaryDirectory() as directory:
        suffix = ".nii.gz" if file.filename.lower().endswith(".nii.gz") else ".nii"
        path = Path(directory) / ("study" + suffix)
        size = 0
        with path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 256 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="File exceeds 256 MiB")
                output.write(chunk)
        try:
            image = nib.as_closest_canonical(nib.load(str(path), mmap=False, keep_file_open=False))
            if len(image.shape) not in (3, 4):
                raise ValueError("Only 3D or 4D NIfTI volumes are supported")
            original_shape = list(image.shape)
            volume_count = image.shape[3] if len(image.shape) == 4 else 1
            if volume_index >= volume_count:
                raise HTTPException(status_code=422, detail="Volume index out of range")
            if np.prod(image.shape[:3]) > 100_000_000:
                raise ValueError("3D volume exceeds 100M voxels")
            if np.prod(image.shape) * 4 > CACHE_MAX_BYTES:
                raise HTTPException(status_code=413, detail="Decoded MRI exceeds 256 MiB memory limit")
            all_data = image.get_fdata(dtype=np.float32, caching="unchanged")
            image.uncache()
            volume = all_data[..., volume_index].copy() if len(image.shape) == 4 else all_data
            del all_data
            finite = volume[np.isfinite(volume)]
            if finite.size == 0:
                raise ValueError("No finite intensities")
            lower, upper = map(float, np.percentile(finite, [1, 99]))
            if upper <= lower:
                upper = lower + 1
            volume = np.nan_to_num(volume, nan=lower, posinf=upper, neginf=lower, copy=False)
            indices = {name: int(volume.shape[axis] // 2) for axis, name in enumerate(("sagittal", "coronal", "axial"))}
            previews = {name: preview_image(volume, axis, indices[name], lower, upper)
                        for axis, name in enumerate(indices)}
            study_id = secrets.token_urlsafe(32)
            with _cache_lock:
                _cache.clear()  # Single-study research workstation; bounded memory.
                _cache[study_id] = {"volume": volume, "lower": lower, "upper": upper,
                                    "last_access": time.monotonic(), "mask": None, "spacing": list(map(float, image.header.get_zooms()[:3]))}
            return {
                "study_id": study_id,
                "filename": file.filename,
                "shape": list(volume.shape),
                "original_shape": original_shape,
                "volume_count": volume_count,
                "selected_volume": volume_index,
                "voxel_spacing_mm": list(map(float, image.header.get_zooms()[:3])),
                "orientation": "".join(nib.aff2axcodes(image.affine)),
                "previews": previews,
                "slice_indices": indices,
                "status": "inspected",
                "notice": "4D input: displaying one selected volume only; sequence identity is not inferred." if volume_count > 1 else None,
                "analysis_available": False,
            }
        except (ValueError, OSError, EOFError, nib.filebasedimages.ImageFileError) as exc:
            raise HTTPException(status_code=422, detail="Invalid or unsupported NIfTI volume") from exc


@app.get("/api/studies/{study_id}/slice/{plane}")
def get_cached_slice(study_id: str, plane: str, index: int = Query(ge=0)):
    axes = {"sagittal": 0, "coronal": 1, "axial": 2}
    if plane not in axes:
        raise HTTPException(status_code=400, detail="Unknown MRI plane")
    item = _get_session(study_id)
    axis = axes[plane]
    if index >= item["volume"].shape[axis]:
        raise HTTPException(status_code=422, detail="Slice index out of range")
    from app.inference import overlay_png
    render = overlay_png(item["volume"], item["mask"], axis, index, item["lower"], item["upper"]) if item.get("mask") is not None else preview_image(item["volume"], axis, index, item["lower"], item["upper"])
    return {"plane": plane, "index": index, "preview": render, "segmentation_overlay": item.get("mask") is not None}


@app.delete("/api/studies/{study_id}")
def release_study(study_id: str):
    with _cache_lock:
        _cache.pop(study_id, None)
    return {"released": True}


@app.get("/api/studies/{study_id}")
def study_session_status(study_id: str):
    """Metadata-only session status; no medical image data is returned."""
    item = _get_session(study_id)
    return {"study_id": study_id, "shape": list(item["volume"].shape),
            "expires_after_inactive_seconds": CACHE_TTL_SECONDS,
            "analysis_available": item.get("mask") is not None}


@app.get("/api/ai/readiness")
def ai_readiness():
    from app.research_pipeline import model_readiness
    return model_readiness()


@app.post("/api/studies/{study_id}/preprocess/summary")
def preprocessing_summary(study_id: str):
    """Research-only intensity statistics; never returns tumor predictions."""
    from app.research_pipeline import preprocess_mri
    item = _get_session(study_id)
    try:
        _, summary = preprocess_mri(item["volume"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "study_id": study_id,
        "shape": list(summary.shape),
        "nonzero_voxels": summary.nonzero_voxels,
        "normalized_mean": round(summary.normalized_mean, 6),
        "normalized_std": round(summary.normalized_std, 6),
        "clipping_percentiles": list(summary.clipping_percentiles),
        "research_only": True,
        "model_loaded": False,
    }


@app.post("/api/studies/{study_id}/ai/segment")
def segment_study(study_id: str, modality: str = Query(default="flair")):
    """Opt-in experimental whole-tumor inference. Only use actual FLAIR input."""
    if modality.lower() != "flair":
        raise HTTPException(status_code=422, detail="This model supports FLAIR only")
    item = _get_session(study_id)
    from app.inference import infer, overlay_png
    try:
        mask = infer(item["volume"])
    except (RuntimeError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    with _cache_lock:
        if _cache.get(study_id) is not item:
            raise HTTPException(status_code=404, detail="Study session expired")
        item["mask"] = mask
    volume_mm3 = float(mask.sum()) * float(__import__("numpy").prod(item["spacing"]))
    indices = {name: int(item["volume"].shape[axis] // 2) for axis, name in enumerate(("sagittal", "coronal", "axial"))}
    previews = {name: overlay_png(item["volume"], mask, axis, indices[name], item["lower"], item["upper"])
                for axis, name in enumerate(indices)}
    return {"study_id": study_id, "task": "binary_whole_tumor", "modality": "FLAIR",
            "threshold": 0.90, "postprocessing": "largest_3d_component",
            "segmented_voxels": int(mask.sum()), "estimated_mask_volume_ml": round(volume_mm3 / 1000, 3),
            "previews": previews, "slice_indices": indices,
            "research_only": True, "notice": "Experimental segmentation, not a diagnosis. Only use FLAIR MRI."}

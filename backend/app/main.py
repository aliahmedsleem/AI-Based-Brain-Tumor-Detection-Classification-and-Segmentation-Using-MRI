from pathlib import Path
from io import BytesIO
import base64
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NeuroScope AI Research API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "research", "model_loaded": False}


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


@app.post("/api/studies/inspect")
async def inspect_study(file: UploadFile = File(...)):
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
            image = nib.as_closest_canonical(nib.load(str(path)))
            if len(image.shape) != 3:
                raise ValueError("Only 3D MRI volumes supported")
            if np.prod(image.shape) > 100_000_000:
                raise ValueError("Volume exceeds safe preview size")
            volume = image.get_fdata(dtype=np.float32)
            finite = volume[np.isfinite(volume)]
            if finite.size == 0:
                raise ValueError("No finite intensities")
            lower, upper = np.percentile(finite, [1, 99])
            if upper <= lower:
                upper = lower + 1
            volume = np.nan_to_num(volume, nan=float(lower), posinf=float(upper), neginf=float(lower))
            indices = [int(n // 2) for n in image.shape]
            previews = {
                name: preview_image(volume, axis, indices[axis], lower, upper)
                for axis, name in enumerate(["sagittal", "coronal", "axial"])
            }
            return {
                "filename": file.filename,
                "shape": list(image.shape),
                "voxel_spacing_mm": list(map(float, image.header.get_zooms()[:3])),
                "orientation": "".join(nib.aff2axcodes(image.affine)),
                "previews": previews,
                "slice_indices": dict(zip(["sagittal", "coronal", "axial"], indices)),
                "status": "inspected",
                "analysis_available": False,
            }
        except (ValueError, OSError, EOFError, nib.filebasedimages.ImageFileError) as exc:
            raise HTTPException(status_code=422, detail="Invalid or unsupported 3D NIfTI volume") from exc

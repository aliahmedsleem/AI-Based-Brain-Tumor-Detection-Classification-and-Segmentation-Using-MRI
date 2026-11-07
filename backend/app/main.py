from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NeuroScope AI Research API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "research", "model_loaded": False}


@app.post("/api/studies/inspect")
async def inspect_study(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Only NIfTI .nii or .nii.gz files are supported in this phase",
        )

    import nibabel as nib

    with tempfile.TemporaryDirectory() as directory:
        suffix = ".nii.gz" if file.filename.lower().endswith(".nii.gz") else ".nii"
        path = Path(directory) / f"study{suffix}"
        size = 0
        with path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 256 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="File exceeds 256 MiB limit")
                output.write(chunk)
        try:
            image = nib.load(str(path))
            if len(image.shape) < 3:
                raise ValueError("Expected at least three dimensions")
            spacing = list(map(float, image.header.get_zooms()[:3]))
            return {
                "filename": file.filename,
                "shape": list(image.shape),
                "voxel_spacing_mm": spacing,
                "status": "inspected",
                "analysis_available": False,
            }
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail="Invalid or unsupported NIfTI image"
            ) from exc

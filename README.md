# NeuroScope AI — Brain Tumor MRI Research Workstation

> Phase 1 research prototype. **Not a medical device or diagnostic tool.**
> No trained AI model is included. No tumor detection, segmentation or classification is performed yet.

## Architecture
- Frontend: React 19, TypeScript, Vite, Lucide icons
- Backend: FastAPI, NiBabel, NumPy
- Planned: PyTorch/MONAI, PostgreSQL, NIfTI/DICOM viewer, 3D segmentation, volumetry and PDF reports

## Run on Windows
Install Python 3.12, Node.js 22 and Git. Clone this repository:

```powershell
git clone https://github.com/aliahmedsleem/AI-Based-Brain-Tumor-Detection-Classification-and-Segmentation-Using-MRI.git
cd AI-Based-Brain-Tumor-Detection-Classification-and-Segmentation-Using-MRI
```

Backend (PowerShell terminal 1):
```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Frontend (PowerShell terminal 2, from project root):
```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 . API documentation: http://localhost:8000/docs .

## Current functionality
- Responsive research dashboard, dark sidebar and structured workspace
- NIfTI (.nii/.nii.gz) metadata inspection with dimension and physical voxel spacing
- 256 MiB upload limit and friendly error states
- Explicitly marked unavailable modules; no fabricated medical results
- GitHub Actions checks: Python syntax and frontend build

## Roadmap
1. UI/UX design system + MRI metadata ingestion — **in progress**
2. Validated medical imaging viewer, safe study storage, authentication
3. Research datasets, patient-level splitting, preprocessing and reproducibility
4. Classification and segmentation model training and independent evaluation
5. Tumor localization, calibrated uncertainty, volumetry, 3D visualization
6. Research reports, external validation and security/accessibility testing

## Privacy and scientific validation
Only use de-identified, licensed research data. Never commit medical images, credentials, or private files. Correctly handle NIfTI affine orientation, voxel spacing and modality identity when calculating location and volume. MRI-based predictions do not definitively determine histopathological tumor type.

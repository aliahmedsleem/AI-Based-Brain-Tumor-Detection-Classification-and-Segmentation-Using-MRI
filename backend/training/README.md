# NeuroScope AI — experimental 3D segmentation training

**Research only. No pretrained weights or clinical performance claims are included.**

This folder provides a reproducible starting point for *binary whole-tumor* segmentation with a 3D UNet. It is not a validated BraTS challenge submission. Training is intentionally opt-in and does not run in the web API.

## Input manifest

Create a JSON file containing an array of records:

```json
[
  {"patient_id":"case_001","image":"C:/data/case_001_flair.nii.gz","label":"C:/data/case_001_seg.nii.gz"},
  {"patient_id":"case_002","image":"C:/data/case_002_flair.nii.gz","label":"C:/data/case_002_seg.nii.gz"}
]
```

Use de-identified data only. Paths can be absolute or relative to the manifest's directory. Each record must have a unique patient_id. Do not commit image data or patient identifiers. All images must be 3D, single-modality NIfTI, with labels spatially aligned to their corresponding images. BraTS labels are converted to binary tumor-present vs background (nonzero vs zero); this discards enhancing/core/edema distinctions. Label semantics must be confirmed against the dataset's documentation.

## Setup and training (PowerShell)

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install -r training/requirements.txt
.\.venv\Scripts\python.exe -m training.train --manifest "C:\data\research_manifest.json" --out "C:\data\neuroscope_runs\run01" --epochs 20
```

For a quick CPU smoke test, use `--epochs 1 --device cpu --roi-size 64` on a small, properly licensed dataset. GPU training is strongly recommended. Defaults use 96³ patches and batch size 1; memory use varies substantially by hardware. The script saves the split and best validation checkpoint. Do not evaluate test cases during training.

## Validation boundaries

- Patient-level, seeded split (70% train / 15% validation / 15% test).
- **No inference endpoint** until independently evaluated weights, correct modality mapping and test results are available.
- This baseline assumes image/label voxel alignment and does not resample or check affine equivalence beyond a strict equality check; preprocess/resample in a separate validated pipeline if necessary.
- The test partition is reserved and never used by the trainer.
- Dice on the validation partition is for development only; it is not clinical accuracy.


## Windows installation troubleshooting
If pip reports a missing header under `torch/include/ATen/...`, Windows path length is a likely cause. Keep the training environment at a short path (for example `C:\\NeuroScopeAI\\train-env`) rather than under the deeply nested repository. The API environment can remain unchanged. If torch imports but reports missing DLLs, install/update the Microsoft Visual C++ x64 Redistributable and reinstall torch in the clean environment. Always verify `torch.cuda.is_available()` before selecting `--device cuda`.

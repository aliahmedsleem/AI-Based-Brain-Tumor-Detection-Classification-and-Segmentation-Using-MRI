# Anatomical localization — research workflow (v0.16)

NeuroScope does **not** identify a lobe from a tumor centroid. Anatomical
localization requires an MRI template, a labeled atlas defined in that
template's physical space, and reviewed image registration.

## Prerequisites

1. Obtain a **licensed, documented** anatomical lobe atlas and its matching
   MRI template from a trusted research source. This repository does **not**
   bundle or silently download an atlas. Check the atlas license and label
   descriptions before use.
2. Use a template with a comparable MRI contrast (FLAIR recommended) and
   preprocessing to the subject. Registration on pathology-distorted brains
   may be unreliable.
3. Install the optional registration dependency in the Python environment
   that already has the API dependencies:

   `python -m pip install SimpleITK`

## Register template and labels to the subject

From `backend`:

```powershell
C:\NeuroScopeAI\train-env\Scripts\python.exe -m app.register_atlas `
  --subject "C:\path\BraTS2021_01065_flair.nii.gz" `
  --template "C:\path\template_flair.nii.gz" `
  --labels "C:\path\atlas_lobes.nii.gz" `
  --out-dir "C:\NeuroScopeAI\atlas_registration\case01065"
```

Outputs:

- `registered_atlas.nii.gz`: nearest-neighbor labels on the subject MRI grid
- `registered_template.nii.gz`: transformed template intensity volume
- `template_to_subject.tfm`: registration transform
- `registration_qc_axial.png`: **one preliminary QA view only**
- `registration_report.json`: method, metric, warnings

**Important:** This is a **rigid registration baseline**, not a clinically
validated deformable pipeline. A low registration metric is not evidence of
correct lobe alignment. Examine overlays in **all three planes** with 3D
Slicer and check label-to-lobe definitions, laterality, and tumor mass effect.
If QA fails, **do not use the atlas localization**.

## Report lobe overlap

1. Run FLAIR segmentation in the web application.
2. In Reports → Anatomical Atlas Localization, upload the reviewed
   `registered_atlas.nii.gz`.
3. Supply the atlas label mapping from the **actual atlas documentation**,
   e.g. (illustrative codes only; do not use unless the atlas uses them):

```json
{
  "10": {"lobe": "frontal", "hemisphere": "Left"},
  "20": {"lobe": "temporal", "hemisphere": "Right"}
}
```

4. Attest that registration and mapping were reviewed; calculate overlap.
5. Refresh the report. The result is labeled **atlas-estimated** and includes
   each mapped region's fraction of the predicted mask and total atlas coverage.

**No clinical diagnosis:** MRI segmentation, lobe overlap and bounding-box
extent must be independently reviewed by a neuroradiologist. This software
does not determine tumor grade, pathology, or clinical treatment.

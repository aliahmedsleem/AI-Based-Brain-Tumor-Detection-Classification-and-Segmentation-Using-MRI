"""Build a de-identified BraTS 2021 manifest and audit local MRI cases.

Usage: python -m training.prepare_brats --data-root "C:/.../BraTS2021_Training_Data" --out "C:/.../research_manifest.json"
No NIfTI data is uploaded, copied, or modified.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

MODALITIES = ("flair", "t1", "t1ce", "t2", "seg")


def prepare(data_root: Path, out: Path, modality: str = "flair", strict: bool = False):
    if modality not in MODALITIES[:-1]:
        raise ValueError("modality must be flair, t1, t1ce, or t2")
    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Data folder not found: {root}")
    if out.resolve().is_relative_to(root):
        raise ValueError("Write manifest outside the MRI dataset directory")
    cases = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("BraTS2021_"))
    if not cases:
        raise ValueError("No BraTS2021_* case directories found. Check --data-root.")
    records, rejected = [], []
    counts = Counter()
    for case in cases:
        matches = {}
        problems = []
        for suffix in MODALITIES:
            candidates = [case / f"{case.name}_{suffix}.nii.gz", case / f"{case.name}_{suffix}.nii"]
            found = [p for p in candidates if p.is_file()]
            if len(found) != 1:
                problems.append(f"{suffix}: expected one file, found {len(found)}")
            else:
                matches[suffix] = found[0]
                counts[suffix] += 1
        if problems:
            rejected.append({"case": case.name, "issues": problems})
            continue
        records.append({
            "patient_id": case.name,
            "image": str(matches[modality]),
            "label": str(matches["seg"]),
        })
    report = {
        "dataset": "BraTS2021_Training_Data",
        "modality": modality,
        "cases_found": len(cases),
        "valid_cases": len(records),
        "rejected_cases": len(rejected),
        "modality_files_found": dict(counts),
        "rejected": rejected,
        "note": "File presence only; image/label geometry and label values must be checked before training.",
    }
    report_path = out.with_name(out.stem + "_audit.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if strict and rejected:
        raise ValueError(f"{len(rejected)} incomplete cases; audit: {report_path}")
    if len(records) < 7:
        raise ValueError(f"Only {len(records)} valid cases. At least seven required. Audit: {report_path}")
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modality", choices=MODALITIES[:-1], default="flair")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        report, audit = prepare(args.data_root, args.out, args.modality, args.strict)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Cases: {report['cases_found']}; valid: {report['valid_cases']}; incomplete: {report['rejected_cases']}")
    print(f"Manifest: {args.out.resolve()}")
    print(f"Audit: {audit.resolve()}")


if __name__ == "__main__":
    main()

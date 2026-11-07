"""Validate de-identified training manifests and produce patient-disjoint splits."""
import argparse
import json
import random
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) < 7:
        raise ValueError("Manifest must be a list of at least seven independent patients")
    seen = set()
    result = []
    for entry in raw:
        if not isinstance(entry, dict) or not all(entry.get(k) for k in ("patient_id", "image", "label")):
            raise ValueError("Each record requires patient_id, image and label")
        patient = str(entry["patient_id"])
        if patient in seen:
            raise ValueError(f"Duplicate patient_id: {patient}")
        seen.add(patient)
        files = {}
        for key in ("image", "label"):
            candidate = Path(entry[key])
            candidate = (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
            if not candidate.is_file() or not candidate.name.lower().endswith((".nii", ".nii.gz")):
                raise ValueError(f"Missing/unsupported {key} NIfTI for patient {patient}")
            files[key] = str(candidate)
        result.append({"patient_id": patient, **files})
    return result


def split_patients(records: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    shuffled = sorted(records, key=lambda x: x["patient_id"])
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = max(1, int(n * 0.70))
    val_end = max(train_end + 1, int(n * 0.85))
    if val_end >= n:
        val_end = n - 1
    return {"train": shuffled[:train_end], "val": shuffled[train_end:val_end], "test": shuffled[val_end:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    partitions = split_patients(load_manifest(args.manifest), args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(partitions, indent=2), encoding="utf-8")
    print({name: len(rows) for name, rows in partitions.items()})


if __name__ == "__main__":
    main()

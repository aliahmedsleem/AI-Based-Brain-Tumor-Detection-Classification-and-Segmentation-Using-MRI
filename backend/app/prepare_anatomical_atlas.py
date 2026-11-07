"""One-command *preparation* of a Harvard-Oxford atlas registration bundle.

Does NOT approve QA or infer a clinical anatomical location.
Usage: python -m app.prepare_anatomical_atlas --subject CASE.nii.gz
       --template TEMPLATE.nii.gz --atlas ATLAS.nii.gz
       --xml HarvardOxford-Cortical.xml --out-dir OUTPUT
"""
import argparse
import json
from pathlib import Path

from app.harvard_oxford_mapping import generate_mapping
from app.register_atlas import register


def prepare(subject, template, atlas, xml, out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Use a new empty output directory for each case")
    mapping_report = generate_mapping(Path(xml))
    if not mapping_report["mapping"]:
        raise ValueError("No valid lobe mappings in supplied atlas XML")
    registration_report = register(subject, template, atlas, out_dir)
    mapping_path = out_dir / "atlas_mapping.json"
    mapping_path.write_text(
        json.dumps(mapping_report["mapping"], indent=2), encoding="utf-8")
    provenance_path = out_dir / "atlas_mapping_provenance.json"
    provenance_path.write_text(
        json.dumps(mapping_report, indent=2), encoding="utf-8")
    manifest = {
        "status": "requires_manual_registration_qa",
        "atlas_labels": "registered_atlas.nii.gz",
        "atlas_mapping": mapping_path.name,
        "atlas_mapping_provenance": provenance_path.name,
        "registration_report": "registration_report.json",
        "qc_images": registration_report["qc_images"],
        "required_action": "Review the registration in all three planes and verify the atlas XML and NIfTI correspond to the same atlas before uploading.",
        "caution": "The output is a research preparation bundle, not an automatically verified anatomical localization.",
    }
    (out_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("subject", "template", "atlas", "xml", "out-dir"):
        parser.add_argument("--" + option, required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare(args.subject, args.template, args.atlas, args.xml, args.out_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

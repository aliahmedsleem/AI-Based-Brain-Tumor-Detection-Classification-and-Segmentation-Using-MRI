"""Validate anatomical atlas label definitions before research localization.

For FSL Harvard-Oxford maxprob summary NIfTI images, the integer voxel
label is the XML label index + 1 (not the raw XML index). The standard
non-lateralized cortical atlas does not encode left/right hemisphere.
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

# Conservative, explicit region-name keywords. Ambiguous regions are excluded.
REGIONS = {
    "frontal": ("frontal pole", "frontal orbital cortex", "superior frontal gyrus",
                "middle frontal gyrus", "inferior frontal gyrus", "frontal operculum"),
    "temporal": ("superior temporal gyrus", "middle temporal gyrus",
                 "inferior temporal gyrus", "temporal fusiform", "temporal pole",
                 "planum temporale", "planum polare", "heschl"),
    "parietal": ("superior parietal lobule", "supramarginal gyrus",
                 "angular gyrus", "postcentral gyrus", "parietal operculum",
                 "precuneous cortex"),
    "occipital": ("occipital pole", "occipital fusiform", "lateral occipital cortex",
                  "cuneal cortex", "supracalcarine cortex", "intracalcarine cortex"),
    "insula": ("insular cortex",),
}


def generate_mapping(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    name = root.findtext("./header/name", default="")
    if "Harvard-Oxford Cortical" not in name:
        raise ValueError("This mapper only supports the Harvard-Oxford Cortical XML")
    atlas_type = root.findtext("./header/type", default="").strip().lower()
    if atlas_type and atlas_type not in ("probabilistic", "label"):
        raise ValueError("Unsupported atlas type")
    # FSL probabilistic summary/maxprob images encode XML index + 1.
    # For a true Label atlas, the XML index is the label value itself.
    offset = 0 if atlas_type == "label" else 1
    mapping = {}
    excluded = []
    for label in root.findall("./data/label"):
        code = int(label.attrib["index"]) + offset
        text = " ".join((label.text or "").lower().split())
        matches = [lobe for lobe, words in REGIONS.items() if any(word in text for word in words)]
        if code <= 0:
            excluded.append({"label": code, "name": label.text, "reason": "background label"})
        elif len(matches) == 1:
            # Hemisphere intentionally unknown; standard cortical summary
            # labels are not left/right separated.
            mapping[str(code)] = {"lobe": matches[0], "hemisphere": "Unknown"}
        else:
            excluded.append({"label": code, "name": label.text, "reason": "ambiguous or not mapped"})
    return {"atlas": name, "atlas_type": atlas_type or "assumed_probabilistic_summary",
            "label_value_offset": offset, "source_xml": str(xml_path),
            "mapping": mapping, "excluded": excluded,
            "warning": "This atlas is not lateralized. Hemisphere cannot be derived from these labels."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = generate_mapping(args.xml)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Mapped {len(report['mapping'])} labels; excluded {len(report['excluded'])}.")
    print("Hemisphere is Unknown for this atlas; never claim left/right from these labels.")


if __name__ == "__main__":
    main()

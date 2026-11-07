"""Unit tests for conservative Harvard-Oxford label mapping."""
import tempfile
import unittest
from pathlib import Path

from app.harvard_oxford_mapping import generate_mapping


class HarvardOxfordMappingTests(unittest.TestCase):
    def test_xml_indices_and_excluded_regions(self):
        xml = """<atlas><header><name>Harvard-Oxford Cortical Structural Atlas</name></header>
        <data>
          <label index="0">Frontal Pole</label>
          <label index="1">Temporal Pole</label>
          <label index="2">Cerebral White Matter</label>
        </data></atlas>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atlas.xml"
            path.write_text(xml, encoding="utf-8")
            report = generate_mapping(path)
        self.assertEqual(report["mapping"]["1"], {"lobe": "frontal", "hemisphere": "Unknown"})
        self.assertEqual(report["mapping"]["2"], {"lobe": "temporal", "hemisphere": "Unknown"})
        self.assertNotIn("3", report["mapping"])
        self.assertEqual(len(report["excluded"]), 1)

    def test_reject_wrong_atlas(self):
        xml = "<atlas><header><name>Other Atlas</name></header><data /></atlas>"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.xml"
            path.write_text(xml, encoding="utf-8")
            with self.assertRaises(ValueError):
                generate_mapping(path)


if __name__ == "__main__":
    unittest.main()

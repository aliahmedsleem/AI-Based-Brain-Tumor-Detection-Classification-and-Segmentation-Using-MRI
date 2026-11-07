import unittest
import numpy as np
from app.anatomical_localization import atlas_overlap


class AtlasOverlapTests(unittest.TestCase):
    def test_fraction_by_lobe(self):
        mask = np.ones((2, 2, 2), dtype=bool)
        atlas = np.zeros((2, 2, 2), dtype=np.int16)
        atlas[:1] = 10
        atlas[1:] = 20
        mapping = {"10": {"lobe": "frontal", "hemisphere": "Left"},
                   "20": {"lobe": "temporal", "hemisphere": "Right"}}
        result = atlas_overlap(mask, atlas, mapping)
        self.assertEqual(result["coverage_fraction"], 1)
        self.assertEqual(len(result["regions"]), 2)
        self.assertEqual(result["regions"][0]["fraction_of_predicted_mask"], 0.5)

    def test_missing_registration_labels_report_low_coverage(self):
        mask = np.ones((2, 2, 2), dtype=bool)
        atlas = np.zeros((2, 2, 2), dtype=np.int16)
        result = atlas_overlap(mask, atlas, {"10": {"lobe": "frontal", "hemisphere": "Left"}})
        self.assertEqual(result["coverage_fraction"], 0)
        self.assertEqual(result["regions"], [])

    def test_reject_bad_shape(self):
        with self.assertRaises(ValueError):
            atlas_overlap(np.zeros((2, 2, 2)), np.zeros((2, 2, 3)), {})


if __name__ == "__main__":
    unittest.main()

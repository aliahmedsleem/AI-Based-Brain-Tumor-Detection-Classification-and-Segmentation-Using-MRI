import unittest
import numpy as np
from app.mask_measurements import describe_mask, compare_masks


class MaskMeasurementsTests(unittest.TestCase):
    def test_volume_and_extent(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:5, 3:7, 4:6] = 1
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        result = describe_mask(mask, affine)
        self.assertEqual(result["segmented_voxels"], 24)
        self.assertEqual(result["estimated_mask_volume_ml"], 0.576)
        self.assertEqual(result["bounding_box_extent_mm"], [6.0, 12.0, 8.0])
        self.assertIsNone(result["anatomical_region"])

    def test_perfect_overlap(self):
        a = np.array([0, 1, 1, 0], dtype=bool)
        result = compare_masks(a, a)
        self.assertEqual(result["dice"], 1)
        self.assertEqual(result["precision"], 1)
        self.assertEqual(result["recall"], 1)

    def test_partial_overlap(self):
        a = np.array([1, 1, 0, 0], dtype=bool)
        b = np.array([0, 1, 1, 0], dtype=bool)
        self.assertEqual(compare_masks(a, b)["dice"], 0.5)

    def test_empty_masks(self):
        a = np.zeros((3, 3, 3), dtype=bool)
        self.assertEqual(describe_mask(a, np.eye(4))["segmented_voxels"], 0)
        self.assertEqual(compare_masks(a, a)["dice"], 1)


if __name__ == "__main__":
    unittest.main()

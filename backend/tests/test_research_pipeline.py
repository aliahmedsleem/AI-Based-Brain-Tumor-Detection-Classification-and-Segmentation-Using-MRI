import unittest
import numpy as np
from app.research_pipeline import model_readiness, preprocess_mri


class ResearchPipelineTests(unittest.TestCase):
    def test_normalization_excludes_background(self):
        rng = np.random.default_rng(42)
        data = np.zeros((12, 12, 12), dtype=np.float32)
        data[2:10, 2:10, 2:10] = rng.normal(100, 20, (8, 8, 8))
        result, summary = preprocess_mri(data)
        self.assertEqual(result.shape, data.shape)
        self.assertTrue(np.all(result[data == 0] == 0))
        self.assertAlmostEqual(summary.normalized_mean, 0.0, places=4)
        self.assertAlmostEqual(summary.normalized_std, 1.0, places=4)

    def test_empty_volume_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_mri(np.zeros((4, 4, 4), dtype=np.float32))

    def test_no_fake_inference(self):
        status = model_readiness()
        self.assertFalse(status["model_loaded"])
        self.assertFalse(status["segmentation_available"])


if __name__ == "__main__":
    unittest.main()
